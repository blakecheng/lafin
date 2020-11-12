import datetime
import os
import time
import timeit
import numpy as np
import torch as th
from src.stylegan2 import BaseNetwork,image_noise
from functools import partial
from math import floor, log2
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DisFinalBlock(th.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels):
        """
        constructor of the class
        :param in_channels: number of input channels
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        # modules required:
        self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

        # final conv layer emulates a fully connected layer
        self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, dilation=1):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        from torch.nn import AvgPool2d, LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # convolutional modules
        self.conv_1 = Conv2d(in_channels, in_channels, (3, 3),
                             dilation=dilation, padding=dilation, bias=True)
        self.conv_2 = Conv2d(in_channels, out_channels, (3, 3),
                             dilation=dilation, padding=dilation, bias=True)
        self.downSampler = AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y

class MinibatchStdDev(th.nn.Module):
    def __init__(self, averaging='all'):
        """
        constructor for the class
        :param averaging: the averaging mode used for calculating the MinibatchStdDev
        """
        super().__init__()

        # lower case the passed parameter
        self.averaging = averaging.lower()

        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in \
                   ['all', 'flat', 'spatial', 'none', 'gpool'], \
                   'Invalid averaging mode %s' % self.averaging

        # calculate the std_dev in such a way that it doesn't result in 0
        # otherwise 0 norm operation's gradient is nan
        self.adjusted_std = lambda x, **kwargs: th.sqrt(
            th.mean((x - th.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        """
        forward pass of the Layer
        :param x: input
        :return: y => output
        """
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)

        # compute the std's over the minibatch
        vals = self.adjusted_std(x, dim=0, keepdim=True)

        # perform averaging
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = th.mean(vals, dim=1, keepdim=True)

        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = th.mean(th.mean(vals, 2, keepdim=True), 3, keepdim=True)

        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]

        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = th.mean(th.mean(th.mean(x, 2, keepdim=True),
                                       3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = th.FloatTensor([self.adjusted_std(x)])

        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] /
                             self.n, self.shape[2], self.shape[3])
            vals = th.mean(vals, 0, keepdim=True).view(1, self.n, 1, 1)

        # spatial replication of the computed statistic
        vals = vals.expand(*target_shape)

        # concatenate the constant feature map to the input
        y = th.cat([x, vals], 1)

        # return the computed value
        return y
        
from torch import tanh

class msg_stylegan2_lm_id_G(BaseNetwork):
    def __init__(self, image_size, latent_dim, style_depth = 8, network_capacity = 16, num_layers = None, transparent = False, attn_layers = [], fmap_max = 512, num_init_filters=1,arc_eval = True):
        super().__init__()
        from .face_modules.model import Backbone
        from .stylegan2 import DiscriminatorBlock,GeneratorBlock
        arcface = Backbone(50, 0.6, 'ir_se').cuda()
        
        
        arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth'), strict=False)
        self.arcface = arcface
        self.arc_eval = arc_eval
        if self.arc_eval == True:
            self.arcface.eval()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.style_depth = style_depth
        if num_layers == None:
            self.num_layers = int(log2(image_size) - 1)
        else:
            self.num_layers = num_layers
        
        ## stylegan e
        
        
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
        
        if self.arc_eval == True:
            with torch.no_grad():
                self.arcface.eval()
                resize_img = F.interpolate(ref_image, [112, 112], mode='bilinear', align_corners=True)
                zid, X_feats = self.arcface(resize_img)
                styles = zid.view(batch_size, 1, self.latent_dim)
        else:
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

        outputs = []
        rgb = None
        for style, block, attn in zip(styles, self.g_blocks, self.g_attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)
            outputs.append(rgb)
        return rgb,outputs


class msg_stylegan2_lm_id_D(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512, dilation=1, use_spectral_norm=True):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param dilation: amount of dilation to be applied to
                         the 3x3 convolutional blocks of the discriminator
        :param use_spectral_norm: whether to use spectral_normalization
        """
        from torch.nn import ModuleList
        # from MSG_GAN.CustomLayers import DisGeneralConvBlock, DisFinalBlock
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.depth = depth
        self.feature_size = feature_size
        self.spectral_norm_mode = None
        self.dilation = dilation

        # create the fromRGB layers for various inputs:
        def from_rgb(out_channels):
            return Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList([from_rgb(self.feature_size // 2)])

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([DisFinalBlock(self.feature_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    dilation=dilation
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            dilation=dilation)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 1] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()

    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv_1)
            remove_spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 1](inputs[self.depth - 1])
        y = self.layers[self.depth - 1](y)
        for x, block, converter in \
                zip(reversed(inputs[:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        return y
