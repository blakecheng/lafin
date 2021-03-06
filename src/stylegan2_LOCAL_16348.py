import os
import sys
import math
import fire
import json
from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from adamp import AdamP
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from vector_quantize_pytorch import VectorQuantize
from linear_attention_transformer import ImageLinearAttention

from PIL import Image
from pathlib import Path

from .res_unet import MultiScaleResUNet

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# constants

EXTS = ['jpg', 'jpeg', 'png']
EPS = 1e-8

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# helpers

def default(value, d):
    return d if value is None else value

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size()).cuda(),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()

def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]

def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# dataset

def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, num_channels):
        self.num_channels = num_channels
    def __call__(self, tensor):
        return tensor.expand(self.num_channels, -1, -1)

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(num_channels))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# augmentations

def random_float(lo, hi):
    return lo + (hi - lo) * random()

def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random() * delta)
    w_delta = int(random() * delta)
    cropped = tensor[:, :, h_delta:(h_delta + new_width), w_delta:(w_delta + new_width)].clone()
    return F.interpolate(cropped, size=(h, h), mode='bilinear')

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., detach = False):
        if random() < prob:
            random_scale = random_float(0.5, 0.9)
            images = random_hflip(images, prob=0.5)
            images = random_crop_and_resize(images, scale = random_scale)

        if detach:
            images.detach_()

        return self.D(images)

# stylegan2 classes

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

        ## (512,4,4),None,(1,512),(1,256,256,1)
        ## (512,4,4),(3,8,8),...,...
        ## ...
    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1)) 
        ## INPUT: (256,256,1) output: (4, 4, 512)-> (512,4,4)
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1)) # 

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
 
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb
        ## (512,4,4),(3,8,8)
        ## (512,8,8),(3,16,16)
        ## (256, 16, 16), (3, 32, 32)
        ## (128, 32, 32), (3, 64, 64)
        ## (64, 64, 64), (3, 128, 128)
        ## (32, 128, 128),(3, 256, 256)
        ## (16, 256, 256), (3, 256, 256)
        

class Reenactment_DecoderBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    ## iatt:
    ## (512,4,4)
    ## (512,8,8)
    ## (256, 16, 16)
    ## (128, 32, 32)
    ## (64, 64, 64)
    ## (32, 128, 128)
    ## (16, 256, 256)
    def forward(self, x, prev_rgb, istyle, iatt):
        if self.upsample is not None:
            x = self.upsample(x)

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + iatt)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + iatt)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb
    

class Reenactment_EncoderBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = x + res
        return x
    
class Reenactment_Encoder(nn.Module):
    def __init__(self, image_size = 256 ,num_init_filters=3, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        if num_layers == None:
            num_layers = int(log2(image_size) - 1)
            self.num_layers = num_layers
        else:
            self.num_layers = num_layers
         
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers)]
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters)) 
        print("encoder:",filters)
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)
            not_first = ind != 0
            block = Reenactment_EncoderBlock(in_chan, out_chan, not_first)
            blocks.append(block)
        
#         blocks.append(nn.Conv2d(filters[-1], filters[-1], 3, padding = 1, stride = 2))
        self.e_blocks = torch.nn.ModuleList(blocks)
        
    def forward(self, x):
        iatts = []
        for block in self.e_blocks:
            x = block(x)
            iatts.append(x)
        return iatts
    

class Reenactment_Decoder(nn.Module):
    def __init__(self, latent_dim = 512 ,image_size = 256 , style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        if num_layers == None:
            num_layers = int(log2(image_size))
            self.num_layers = num_layers
        else:
            self.num_layers = num_layers
         
        filters = [(network_capacity) * (2 ** (i)) for i in range(num_layers)[::-1]]
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters)) 
        print("dncoder:",filters)
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        
        blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 2)
            block = Reenactment_DecoderBlock(
                latent_dim,
                in_chan, 
                out_chan, 
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent)
            blocks.append(block)
        self.d_blocks = torch.nn.ModuleList(blocks)
        
    def forward(self,x,styles,re_iatts):
        rgb = None
        rgbs = []
        for block,style,iatt in zip(self.d_blocks,styles, re_iatts):
            x, rgb = block(x, rgb, style, iatt)
            rgbs.append(rgb)
        return rgb,rgbs


class Reenactment_Generator(nn.Module):
     def __init__(self,latent_dim = 512, image_size = 256 ,num_init_filters=3, network_capacity = 16, 
                  num_layers = None, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()   
        self.encoder = Reenactment_Encoder(image_size = image_size ,
                                           num_init_filters=num_init_filters, 
                                           network_capacity = network_capacity,
                                           num_layers = num_layers, 
                                           transparent = transparent, 
                                           attn_layers = attn_layers, 
                                           no_const = no_const, 
                                           fmap_max = fmap_max)
        self.decoder = Reenactment_Decoder(latent_dim = latent_dim ,
                                           image_size = image_size , 
                                           network_capacity = network_capacity, 
                                           num_layers = num_layers, 
                                           transparent = transparent, 
                                           attn_layers = attn_layers, 
                                           no_const = no_const, 
                                           fmap_max = fmap_max
                                          )
    
        self.initial_block = nn.Parameter(torch.randn((1, fmap_max, 4, 4)))
    
     def forward(self,x,styles):
        iatts = self.encoder(x)
        re_iatts = iatts[::-1]
        batch_size = styles[0].shape[0]
        init_x = self.initial_block.expand(batch_size, -1, -1, -1)
        output, rgbs = self.decoder(init_x,styles,re_iatts)
        return output, rgbs, iatts


class stylegan_base_ae(nn.Module):
     def __init__(self,latent_dim = 512, image_size = 256 ,num_init_filters=3, network_capacity = 16, 
                  num_layers = None, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()   
        self.decoder = Reenactment_Decoder(latent_dim = latent_dim ,
                                           image_size = image_size , 
                                           network_capacity = network_capacity, 
                                           num_layers = num_layers, 
                                           transparent = transparent, 
                                           attn_layers = attn_layers, 
                                           no_const = no_const, 
                                           fmap_max = fmap_max
                                          )
    
        self.initial_block = nn.Parameter(torch.randn((1, fmap_max, 4, 4)))
    
     def forward(self,x,landmarks):
        id_latent = self.get_id_latent(x)
        lm_latent = self.get_lm_latent(landmarks)
        style = torch.cat((id_latent,lm_latent),dim=1)
        styles = [style for i in range(self.depth)]
        
        batch_size = styles[0].shape[0]
        init_x = self.initial_block.expand(batch_size, -1, -1, -1)
        output, rgbs = self.decoder(init_x,styles,noise)
        return output, rgbs, iatts
        

class stylegan_base_facereenactment(nn.Module):
    def __init__(self,image_size=256, fmap_max= 512,latent_dim= 1024):
        super().__init__()
        
        from .face_modules.model import Backbone
        arcface = Backbone(50, 0.6, 'ir_se')
        arcface.eval()
        arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth'), strict=False)
        self.arcface = arcface
        self.arcface.eval()
        
        self.latent_dim = latent_dim
        
        self.generator = Reenactment_Generator(image_size = image_size,
                                               fmap_max=fmap_max,
                                               num_init_filters=4,
                                              latent_dim =latent_dim)
        self.depth = int(log2(image_size))-1
        from .oneshot_facereenactment import Normal_Encoder
        self.landmark_encoder = Normal_Encoder(1,3)
    
    def get_id_latent(self,Y):
        batch_size = Y.shape[0]
        with torch.no_grad():
            self.arcface.eval()
            resize_img = F.interpolate(Y, [112, 112], mode='bilinear', align_corners=True)
            zid, X_feats = self.arcface(resize_img)
            id_latent = zid.view(batch_size, int(self.latent_dim/2))
        return id_latent
                                 
    def get_lm_latent(self,lm):
        return self.landmark_encoder(lm)
                                 
    def get_att(self,refx,lmx):
        return self.generator.encoder(torch.cat((refx,lmx),dim=1))
                                 
    def forward(self,landmarks,refimages,ref_landmarks):
        id_latent = self.get_id_latent(refimages)
        lm_latent = self.get_lm_latent(landmarks)
        style = torch.cat((id_latent,lm_latent),dim=1)
        styles = [style for i in range(self.depth)]
        output,rgbs,iatts = self.generator(torch.cat((refimages,ref_landmarks),dim=1),styles)
        return output,rgbs,iatts,id_latent,lm_latent

    


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x
    

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        
    
class stylegan_L2I_Generator(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        ## lafin
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )
        ## concate 
        self.concator = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=4, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True)
        )
        
        ### stylegan
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        
        self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)
            
            self.init_weights()
           
        #noise = torch.FloatTensor(image_size, image_size, 1).uniform_(0., 1.).cuda()
        
        self.num_layers = int(log2(image_size) - 1)
        self.get_latents_fn = noise_list
        self.S = StyleVectorizer(latent_dim, style_depth).cuda()
        
        
    def forward(self, x):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
        style = self.get_latents_fn(batch_size, self.num_layers, self.latent_dim)
        input_noise = image_noise(batch_size, image_size)
        
        w_space = latent_to_w(self.S, style)
        w_styles = styles_def_to_tensor(w_space)
        styles = w_styles

        f_e1 = self.encoder1(x)
        f_e2 = self.encoder2(f_e1)
        f_e3 = self.encoder3(f_e2)
        x = self.concator(f_e3)
        
        #print(x.shape)
        # if self.no_const:
        #     avg_style = styles.mean(dim=1)[:, :, None, None]
        #     x = self.to_initial_block(avg_style)
        # else:
        #     x = self.initial_block.expand(batch_size, -1, -1, -1)

        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb
    
class stylegan_L2I_Generator2(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        ## lafin
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )
        ## concate
        self.concator = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=4, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True)
        )
        ## 4,256,256 in -> 512,2,2 out
        ### stylegan
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        
        self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)
            
            self.init_weights()
           
        #noise = torch.FloatTensor(image_size, image_size, 1).uniform_(0., 1.).cuda()
        
        

        self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))

        
        
        
    def forward(self, x):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
      
        # style固定，noise不固定
        input_noise = image_noise(batch_size, image_size)
        styles = self.single_style.expand(batch_size, -1, -1)

        f_e1 = self.encoder1(x)
        f_e2 = self.encoder2(f_e1)
        f_e3 = self.encoder3(f_e2)
        x = self.concator(f_e3)
        
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb
    




# class Landmark_guided_inpaintor(nn.Module):
#     def __init__(self):
#         super(Landmark_guided_inpaintor, self).__init__()
#         Lencoder = Encoder(in_c=1,depth=3)
        
#     def forward:
#         pass


class ref_guided_inpaintor(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], fmap_max = 512):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        if num_layers == None:
            self.num_layers = int(log2(image_size) - 1)
        else:
            self.num_layers = num_layers
        
        ## stylegan e
        
        num_init_filters = 6
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers)]
        print(filters)
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            #quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            #quantize_blocks.append(quantize_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
        self.lm_encoder = fr_Encoder(in_c=1,depth=3)            
        
        ### stylegan g
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.num_layers)][::-1]
        print(filters)
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)
 
            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
        
        #self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))
        self.style_depth = style_depth
        
    def forward(self,x,landmarks, masks, ref_image=None):
        
        
        batch_size = x.shape[0]
        image_size = self.image_size
        
        if ref_image == None:
            index = torch.randperm(batch_size).cuda()
            ref_image = x[index]
        
        images_masked = (x * (1 - masks).float()) + masks
        x = torch.cat((images_masked, ref_image), dim=1)
        
        
        # style固定，noise不固定
        input_noise = image_noise(batch_size, image_size)
        
        lz = self.lm_encoder(landmarks)
        lz = lz.view(batch_size,1,self.latent_dim)
        styles = lz.expand(-1,self.style_depth,-1)

        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            
            if attn_block is not None:
                x = attn_block(x)


        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb


##################################################################################
## AE correct
class stylegan_L2I_Generator_AE(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], fmap_max = 512 , in_c = 4):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        if num_layers == None:
            self.num_layers = int(log2(image_size) - 1)
        else:
            self.num_layers = num_layers
        
        ## stylegan e
        
        num_init_filters = in_c
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers)]
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            #quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            #quantize_blocks.append(quantize_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
            
        
        ### stylegan g
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
      
        self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))

        
    def forward(self, x, noise_input=None):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
        # style固定，noise不固定
        if noise_input is not None:
            input_noise = noise_input
        else:
            input_noise = image_noise(batch_size, image_size)
            
        styles = self.single_style.expand(batch_size, -1, -1)

        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            
            if attn_block is not None:
                x = attn_block(x)


        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb
    
    



 

class stylegan_L2I_Generator_AE_landmark_in(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], fmap_max = 512):
        super().__init__()
        
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        if num_layers == None:
            self.num_layers = int(log2(image_size) - 1)
        else:
            self.num_layers = num_layers
        
        ## stylegan e
        
        num_init_filters = 1
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers)]
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            #quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            #quantize_blocks.append(quantize_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
            
        
        ### stylegan g
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)

        self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))
        
    def forward(self, x , input_noise=None):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
        # style固定，noise不固定
        if input_noise==None:
            input_noise = image_noise(batch_size, image_size)
        else:
            input_noise = input_noise
            
        styles = self.single_style.expand(batch_size, -1, -1)

        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            
            if attn_block is not None:
                x = attn_block(x)


        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb

#################################################################################################################################################################################

class stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], fmap_max = 512):
        super().__init__()
        from .face_modules.model import Backbone
        arcface = Backbone(50, 0.6, 'ir_se').cuda()
        arcface.eval()
        arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth'), strict=False)
        self.arcface = arcface
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.style_depth = style_depth
        if num_layers == None:
            self.num_layers = int(log2(image_size) - 1)
        else:
            self.num_layers = num_layers
        
        ## stylegan e
        
        num_init_filters = 1
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers)]
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            #quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            #quantize_blocks.append(quantize_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
            
        
        ### stylegan g
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)

        # self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))
        
    def forward(self, x , ref_image,input_noise=None):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
        # style固定，noise不固定
        if input_noise==None:
            input_noise = image_noise(batch_size, image_size)
        else:
            input_noise = input_noise
        
        with torch.no_grad():
            resize_img = F.interpolate(ref_image, [112, 112], mode='bilinear', align_corners=True)
            zid, X_feats = self.arcface(resize_img)
            styles = zid.view(batch_size, 1, self.latent_dim)
            
        styles = styles.expand(batch_size,self.style_depth , self.latent_dim)
        #styles = self.single_style.expand(batch_size, -1, -1)

        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            
            if attn_block is not None:
                x = attn_block(x)

        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb
    

class stylegan_ae_facereenactment(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], fmap_max = 512):
        super().__init__()
        from .face_modules.model import Backbone
        arcface = Backbone(50, 0.6, 'ir_se').cuda(3)
        arcface.eval()
        arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth',map_location="cuda:%s"%torch.cuda.current_device()), strict=False)
        self.arcface = arcface
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.style_depth = style_depth
        if num_layers == None:
            self.num_layers = int(log2(image_size) - 1)
        else:
            self.num_layers = num_layers
        
        ## stylegan e
        
        num_init_filters = 1
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers)]
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            #quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            #quantize_blocks.append(quantize_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
            
        
        ### stylegan g
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
            
        self.ref_encoder = MultiScaleResUNet(in_nc=4, out_nc=1)
        self.init_weights(init_type='xavier')
        # self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))
        
        
    def get_zatt(self,Y,drive_landmark):
        batch_size = Y.shape[0]
        image_size = self.image_size
        with torch.no_grad():
            return self.ref_encoder(torch.cat((Y,drive_landmark), dim=1),None,None,None).view(batch_size,image_size,image_size,1)
        
    def get_style(self,Y):
        batch_size = Y.shape[0]
        with torch.no_grad():
            resize_img = F.interpolate(Y, [112, 112], mode='bilinear', align_corners=True)
            zid, X_feats = self.arcface(resize_img)
            styles = zid.view(batch_size, 1, self.latent_dim)
        return styles
    
    def forward(self, landmarks,refimages,ref_landmarks):
    
        batch_size = refimages.shape[0]
        image_size = self.image_size
        input_noise = self.ref_encoder(torch.cat((refimages,ref_landmarks), dim=1),None,None,None).view(batch_size,image_size,image_size,1)
        
        with torch.no_grad():
            resize_img = F.interpolate(refimages, [112, 112], mode='bilinear', align_corners=True)
            self.arcface.eval()
            zid, X_feats = self.arcface(resize_img)
            styles = zid.view(batch_size, 1, self.latent_dim)
        
        output_styles =  styles
        styles = styles.expand(batch_size,self.style_depth , self.latent_dim)
        
        x = landmarks
        
        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            if attn_block is not None:
                x = attn_block(x)

        styles = styles.transpose(0, 1) #

        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb,input_noise,output_styles
    
    

class stylegan_ae_facereenactment2(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], fmap_max = 512):
        super().__init__()
        from .face_modules.model import Backbone
        arcface = Backbone(50, 0.6, 'ir_se').cuda()
        arcface.eval()
        arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth'), strict=False)
        self.arcface = arcface
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.style_depth = style_depth
        if num_layers == None:
            self.num_layers = int(log2(image_size) - 1)
        else:
            self.num_layers = num_layers
        
        ## stylegan e
        
        num_init_filters = 4
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers)]
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            #quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            #quantize_blocks.append(quantize_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
            
        
        ### stylegan g
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
            
        self.init_weights(init_type='xavier')
        self.single_noise = nn.Parameter(torch.randn(image_size, image_size, 1))
        
        
    def get_zatt(self,Y,drive_landmark):
        x = torch.cat((Y,drive_landmark),dim=1)
        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            if attn_block is not None:
                x = attn_block(x)
        att = x 
        return att
        
    def get_style(self,Y):
        batch_size = Y.shape[0]
        with torch.no_grad():
            resize_img = F.interpolate(Y, [112, 112], mode='bilinear', align_corners=True)
            zid, X_feats = self.arcface(resize_img)
            styles = zid.view(batch_size, 1, self.latent_dim)
        return styles
    
    def forward(self, landmarks,refimages,ref_landmarks):
        batch_size = refimages.shape[0]
        image_size = self.image_size
        #input_noise = self.ref_encoder(torch.cat((refimages,ref_landmarks), dim=1),None,None,None).view(batch_size,image_size,image_size,1)
        input_noise = self.single_noise.expand(batch_size,self.image_size,self.image_size,1)
        
        
        with torch.no_grad():
            resize_img = F.interpolate(refimages, [112, 112], mode='bilinear', align_corners=True)
            self.arcface.eval()
            zid, X_feats = self.arcface(resize_img)
            styles = zid.view(batch_size, 1, self.latent_dim)
        
        output_styles =  styles
        styles = styles.expand(batch_size,self.style_depth , self.latent_dim)
        
        x = torch.cat((refimages,landmarks),dim=1)
        
        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            if attn_block is not None:
                x = attn_block(x)

        styles = styles.transpose(0, 1) #
        att = x 

        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        
        return rgb,att,output_styles

    
########################################################################################################################################################################################################################################################################################################################################

# AE
class stylegan_L2I_Generator3(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)
        
        
        ## stylegan e
        
        num_init_filters = 4
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers)]
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            #quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            #quantize_blocks.append(quantize_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
            
        
        ### stylegan g
        
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
      
        self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))

        
        
        
    def forward(self, x):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
        # style固定，noise不固定
        input_noise = image_noise(batch_size, image_size)
        styles = self.single_style.expand(batch_size, -1, -1)

        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x = block(x)
            
            if attn_block is not None:
                x = attn_block(x)


        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
        
        return rgb
    
## unet
class stylegan_L2I_Generator4(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
    
        self.num_layers = int(log2(image_size) - 1)-1
        
        
        ## stylegan e
        
        num_init_filters = 4
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers+1)]
        print(filters)
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
            
        
        ### stylegan g
        self.g_num_layers = self.num_layers+1
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.g_num_layers)][::-1]
        print(filters)
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        self.funsions = nn.ModuleList([])
        
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.g_num_layers - 1)
            num_layer = self.g_num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
            funsion_block = FunsionBlock(out_chan,int(out_chan/2))
            self.funsions.append(funsion_block)
      
        self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim))) 
        
    def forward(self, x):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
        # style固定，noise不固定
        input_noise = image_noise(batch_size, image_size)
        styles = self.single_style.expand(batch_size, -1, -1)

        x_list = []
        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x_list.append(x)
            x = block(x)
            if attn_block is not None:
                x = attn_block(x)
            
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn,funsion_block in zip(styles, self.g_blocks, self.g_attns, self.funsions):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
            #print(funsion_block,x.shape,[i.shape for i in x_list])
            x = torch.cat((funsion_block(x),x_list.pop()),dim=1) 
        return rgb
    

    
def FunsionBlock(c_in, c_out, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, stride=1, bias=False),
        norm(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )

### lafin
class stylegan_L2I_Generator5(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
    
        self.num_layers = int(log2(image_size) - 1)-1
        
        
        ## stylegan e
        
        num_init_filters = 4
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers+1)]
        print(filters)
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)
            
        
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
        
        self.auto_attn = Auto_Attn(input_nc=filters[-1],norm_layer=None)
            
        
        ### stylegan g
        self.g_num_layers = self.num_layers+1
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.g_num_layers)][::-1]
        print(filters)
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        self.funsions = nn.ModuleList([])
        
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.g_num_layers - 1)
            num_layer = self.g_num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
            funsion_block = FunsionBlock(out_chan,int(out_chan/2))
            self.funsions.append(funsion_block)
      
        self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))
        
             
        
    def forward(self, x , masks , _ , __ ):
    
        batch_size = x.shape[0]
        image_size = self.image_size
        
        # style固定，noise不固定
        input_noise = image_noise(batch_size, image_size)
        styles = self.single_style.expand(batch_size, -1, -1)

        x_list = []
        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x_list.append(x)
            x = block(x)
            if attn_block is not None:
                x = attn_block(x)
        
        smasks = F.interpolate(masks, size=[x.shape[2], x.shape[3]], \
                                    mode='bilinear', align_corners=True)
        x, _ = self.auto_attn(x, x, smasks)
        
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn,funsion_block in zip(styles, self.g_blocks, self.g_attns, self.funsions):
            if attn is not None:
                x = attn(x)
            
            x, rgb = block(x, rgb, style, input_noise)
            #print(funsion_block,x.shape,[i.shape for i in x_list])
            x_tmp = x_list.pop()
            smasks = F.interpolate(masks, size=[x_tmp.shape[2], x_tmp.shape[3]], \
                                    mode='bilinear', align_corners=True)
            x = torch.cat((funsion_block(x),x_tmp*(1-smasks)),dim=1)
            
        return rgb

### dual
class dualnet(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
    
        self.num_layers = int(log2(image_size) - 1)-1
        
        
        ## stylegan e
        
        num_init_filters = 4
        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(self.num_layers+1)]
        print(filters)
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)
            
        
            
        self.e_blocks = nn.ModuleList(blocks)
        self.e_attn_blocks = nn.ModuleList(attn_blocks)
        
            
        
        ### stylegan g
        self.g_num_layers = self.num_layers+1
        
        filters = [network_capacity * (2 ** (i)) for i in range(self.g_num_layers)][::-1]
        print(filters)
        
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        self.g_blocks = nn.ModuleList([])
        self.g_attns = nn.ModuleList([])
        self.funsions = nn.ModuleList([])
        
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.g_num_layers - 1)
            num_layer = self.g_num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.g_attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.g_blocks.append(block)
            funsion_block = FunsionBlock(out_chan,int(out_chan/2))
            self.funsions.append(funsion_block)
      
        self.single_style = nn.Parameter(torch.randn((1,style_depth,latent_dim)))
        
             
        
    def forward(self, x , masks , landmarks):
        
        batch_size = x.shape[0]
        if batch_size<=1:
            print("batch size should be larger than 1")
            exit
        image_size = self.image_size
        
        index = torch.randperm(batch_size).cuda()
        
        ref_x = x[index]
        ref_masks = masks[index]
        ref_landmarks = landmarks[index]
        
        
        # style固定，noise不固定
        input_noise = image_noise(batch_size, image_size)
        styles = self.single_style.expand(batch_size, -1, -1)

        x_list = []
        for (block, attn_block) in zip(self.e_blocks, self.e_attn_blocks):
            x_list.append(x)
            x = block(x)
            if attn_block is not None:
                x = attn_block(x)
        

        
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn,funsion_block in zip(styles, self.g_blocks, self.g_attns, self.funsions):
            if attn is not None:
                x = attn(x)
            
            x, rgb = block(x, rgb, style, input_noise)
            #print(funsion_block,x.shape,[i.shape for i in x_list])
            x_tmp = x_list.pop()
            x = torch.cat((funsion_block(x),x_tmp),dim=1)
            
        return rgb

  

    
    
#### lafin
class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer = nn.InstanceNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc*2), input_nc, input_nc, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (mask) * context_flow + (1-mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret
    
def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""

    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
    
class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            output_nc = output_nc * 4
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc), nonlinearity, self.conv2,)

        self.shortcut = nn.Sequential(self.bypass,)

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out

##########################################
    

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])
        self.funsions = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)
            
            
    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
           

        return rgb
    
    
    

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):
        super().__init__()
        
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        quantize_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        latent_dim = 2 * 2 * filters[-1]

        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if attn_block is not None:
                x = attn_block(x)

            if q_block is not None:
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss

class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)

        self.SE = StyleVectorizer(latent_dim, style_depth)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const)

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = AdamP(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = AdamP(self.D.parameters(), lr = self.lr, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda()
        
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O2')

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x
    

    

class Trainer():
    def __init__(self, name, results_dir, models_dir, image_size, network_capacity, transparent = False, batch_size = 4, mixed_prob = 0.9, gradient_accumulate_every=1, lr = 2e-4, num_workers = None, save_every = 1000, trunc_psi = 0.6, fp16 = False, cl_reg = False, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, aug_prob = 0., dataset_aug_prob = 0., *args, **kwargs):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent
        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const
        self.aug_prob = aug_prob

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.last_cr_loss = 0
        self.q_loss = 0

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr=self.lr, image_size = self.image_size, network_capacity = self.network_capacity, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, *args, **kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        self.loader = cycle(data.DataLoader(self.dataset, num_workers = default(self.num_workers, num_cores), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))

    def train(self):
        assert self.loader is not None, 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        backwards = partial(loss_backwards, self.fp16)

        if self.GAN.D_cl is not None:
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim)
                    noise = image_noise(batch_size, image_size)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader).cuda()
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt)

            self.GAN.D_opt.step()

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in range(self.gradient_accumulate_every):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise)
            fake_output, fake_q_loss = self.GAN.D_aug(generated_images.clone().detach(), detach = True, prob = aug_prob)

            image_batch = next(self.loader).cuda()
            image_batch.requires_grad_()
            real_output, real_q_loss = self.GAN.D_aug(image_batch, prob = aug_prob)

            divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
            disc_loss = divergence

            quantize_loss = (fake_q_loss + real_q_loss).mean()
            self.q_loss = float(quantize_loss.detach().item())

            disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise)
            fake_output, _ = self.GAN.D_aug(generated_images, prob = aug_prob)
            loss = fake_output.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        if self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}')
            self.load(checkpoint_num)
            raise NanException

        # periodically save results

        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
            self.evaluate(floor(self.steps / 1000))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)
            
        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).cuda()
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, save_frames = False):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim)
        latents_high = noise(num_rows ** 2, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        ratios = torch.linspace(0., 8., 100)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        pl_mean = default(self.pl_mean, 0)
        print(f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {pl_mean:.2f} | CR: {self.last_cr_loss:.2f} | Q: {self.q_loss:.2f}')

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        torch.save(self.GAN.state_dict(), self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')
        self.steps = name * self.save_every
        self.GAN.load_state_dict(torch.load(self.model_name(name)))

# %%

# %%
