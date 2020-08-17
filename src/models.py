import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss
from .stylegan2 import stylegan_L2I_Generator,stylegan_L2I_Generator2,stylegan_L2I_Generator3,stylegan_L2I_Generator4,stylegan_L2I_Generator5,stylegan_L2I_Generator_AE
from .stylegan2 import stylegan_L2I_Generator_AE_landmark_in
from .stylegan2 import ref_guided_inpaintor
# from .stylegan2 import dualnet
from .res_unet import MultiScaleResUNet
# from .faceshifter_generator import faceshifter_sin


use_apex = False
## Tip : 开启可以降一点显存，但是速度会变慢
if use_apex == True:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")



class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else: 
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        
        self.is_local = False
        if hasattr(config, 'LOCAL'):
            self.is_local = config.LOCAL
            print("local mode")
        # generator input: [rgb(3) + landmark(1)]
        # discriminator input: [rgb(3)]
        
        self.inpaint_type = None
        if hasattr(config, 'INPAINTOR'):
            self.inpaint_type = config.INPAINTOR
            
        if  self.inpaint_type == "stylegan2":
            print("#####################")
            print("USE stylegan generator!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            generator = stylegan_L2I_Generator(image_size=image_size,latent_dim=latent_dim)
        elif self.inpaint_type == "stylegan2_fixs":
            print("#####################")
            print("USE stylegan generator, fixstyle!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            generator = stylegan_L2I_Generator2(image_size=image_size,latent_dim=latent_dim)
        elif self.inpaint_type == "stylegan2_ae": 
            print("#####################")
            print("USE stylegan generator, AE!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            generator = stylegan_L2I_Generator3(image_size=image_size,latent_dim=latent_dim)
        elif self.inpaint_type == "stylegan2_ae2":
            print("#####################")
            print("USE stylegan generator, AE expand!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            generator = stylegan_L2I_Generator_AE(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
        elif self.inpaint_type == "s2_ae_landmark_in":
            print("#####################")
            print("USE stylegan generator, AE landmark!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            generator = stylegan_L2I_Generator_AE_landmark_in(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
        elif self.inpaint_type == "stylegan2_unet": 
            print("#####################")
            print("USE stylegan generator, unet!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            generator = stylegan_L2I_Generator4(image_size=image_size,latent_dim=latent_dim,fmap_max = 2048)
        elif self.inpaint_type == "lafin_style": 
            print("#####################")
            print("USE stylegan generator, lafin_style!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            generator = stylegan_L2I_Generator5(image_size=image_size,latent_dim=latent_dim,fmap_max = 2048)
        elif self.inpaint_type == "resunet":
            print("#####################")
            print("USE resunet generator!")
            print("#####################\n")
            generator = MultiScaleResUNet(in_nc=4,out_nc=3)
        elif self.inpaint_type == "faceshifter_sin":
            print("#####################")
            print("USE faceshifter generator!")
            print("#####################\n")
            generator = faceshifter_sin()
        elif self.inpaint_type == "ref_guided":
            print("#####################")
            print("USE ref_guided generator!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            generator = ref_guided_inpaintor(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
        else:
            generator = InpaintGenerator()
        
        discriminator = Discriminator(in_channels=4, use_sigmoid=config.GAN_LOSS != 'hinge')
        # if len(config.GPU) > 1:
        #     generator = nn.DataParallel(generator)
        #     discriminator = nn.DataParallel(discriminator)
            # Apex
        if use_apex == True:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
        else:
            # generator = nn.DataParallel(generator, config.GPU)
            # discriminator = nn.DataParallel(discriminator , config.GPU)
            if len(config.GPU) > 1:
                generator = nn.DataParallel(generator)
                discriminator = nn.DataParallel(discriminator)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        self.tv_loss = TVLoss()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        # Apex
        
        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        if use_apex == True:
            generator, gen_optimizer = amp.initialize(generator, self.gen_optimizer, opt_level="O1")
        
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
        if use_apex == True:
            discriminator, dis_optimizer = amp.initialize(discriminator, self.dis_optimizer, opt_level="O1")



    def process(self, images, landmarks, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, landmarks, masks)
    
        
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        if self.is_local == True:
            dis_real, _ = self.discriminator(torch.cat((dis_input_real* masks, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
            dis_fake, _ = self.discriminator(torch.cat((dis_input_fake* masks, landmarks), dim=1))  
        else:
            dis_real, _ = self.discriminator(torch.cat((dis_input_real, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
            dis_fake, _ = self.discriminator(torch.cat((dis_input_fake, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        if self.is_local == True:
            gen_fake, _ = self.discriminator(torch.cat((gen_input_fake* masks, landmarks), dim=1))
        else:
            gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        if self.is_local == True:
            gen_l1_loss = self.l1_loss(outputs* masks, images* masks) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        else:
            gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #generator tv loss
        tv_loss = self.tv_loss(outputs*masks+images*(1-masks))
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, landmarks, masks):
        if "stylegan2" in self.inpaint_type:
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks), dim=1)
            outputs = self.generator(inputs)
        elif self.inpaint_type == "s2_ae_landmark_in":
            outputs = self.generator(landmarks)
        elif "faceshifter" in self.inpaint_type:
            outputs = self.generator(images,landmarks,masks)
        elif "ref_guided" in self.inpaint_type:
            outputs = self.generator(images,landmarks,masks)
        else:
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks), dim=1)
            scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                        mode='bilinear', align_corners=True)
            scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                        mode='bilinear', align_corners=True)
            outputs = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter)                                    # in: [rgb(3) + landmark(1)]
        return outputs

    def backward(self, gen_loss = None, dis_loss = None):
        # Apex
        if use_apex == True:
            with amp.scale_loss(dis_loss, self.dis_optimizer) as scaled_loss: scaled_loss.backward()
        else:
            dis_loss.backward(retain_graph=True)
        self.dis_optimizer.step()

        if use_apex == True:
            with amp.scale_loss(gen_loss, self.gen_optimizer) as scaled_loss: scaled_loss.backward()
        else:
            gen_loss.backward(retain_graph=True)
        self.gen_optimizer.step()

    def backward_joint(self, gen_loss = None, dis_loss = None):
        # Apex
        if use_apex == True:
            with amp.scale_loss(dis_loss, self.dis_optimizer) as scaled_loss: scaled_loss.backward()
        else:
            dis_loss.backward()
        self.dis_optimizer.step()

        #gen_loss.backward()
        if use_apex == True:
            with amp.scale_loss(gen_loss, self.gen_optimizer) as scaled_loss: scaled_loss.backward()
        else:
            gen_loss.backward()
        self.gen_optimizer.step()






from .networks import MobileNetV2

def abs_smooth(x):
    absx = torch.abs(x)
    minx = torch.min(absx,other=torch.ones(absx.shape).cuda())
    r = 0.5 *((absx-1)*minx + absx)
    return r

def loss_landmark_abs(y_true, y_pred):
    loss = torch.mean(abs_smooth(y_pred - y_true))
    return loss

def loss_landmark(landmark_true, landmark_pred, points_num=68):
    landmark_loss = torch.norm((landmark_true-landmark_pred).reshape(-1,points_num*2),2,dim=1,keepdim=True)

    return torch.mean(landmark_loss)

class LandmarkDetectorModel(nn.Module):
    def __init__(self, config):
        super(LandmarkDetectorModel, self).__init__()
        self.mbnet = MobileNetV2(points_num=config.LANDMARK_POINTS)
        self.name = 'landmark_detector'
        self.iteration = 0
        self.config = config

        self.landmark_weights_path = os.path.join(config.PATH, self.name + '.pth')

        if len(config.GPU) > 1:
            self.mbnet = nn.DataParallel(self.mbnet)
            # self.mbnet = nn.DataParallel(self.mbnet, config.GPU)

        self.optimizer = optim.Adam(
            params=self.mbnet.parameters(),
            lr=self.config.LR,
            weight_decay=0.000001
        )


    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'detector': self.mbnet.state_dict()
        }, self.landmark_weights_path)

    def load(self):
        if os.path.exists(self.landmark_weights_path):
            print('Loading landmark detector...')

            if torch.cuda.is_available():
                data = torch.load(self.landmark_weights_path)
            else:
                data = torch.load(self.landmark_weights_path, map_location=lambda storage, loc: storage)

            self.mbnet.load_state_dict(data['detector'])
            self.iteration = data['iteration']
            print('Loading landmark detector complete!')

    def forward(self, images, masks):
        images_masked = images* (1 - masks).float() + masks

        landmark_gen = self.mbnet(images_masked)
        landmark_gen *= self.config.INPUT_SIZE

        return landmark_gen

    def process(self, images, masks, landmark_gt):
        self.iteration += 1
        self.optimizer.zero_grad()

        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape((-1, self.config.LANDMARK_POINTS, 2))
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss", loss.item())]
        return landmark_gen, loss, logs

    def process_aug(self, images, masks, landmark_gt):
        self.optimizer.zero_grad()
        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape(-1,self.config.LANDMARK_POINTS,2)
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss_aug", loss.item())]

        return landmark_gen, loss, logs



    def backward(self, loss):
        loss.backward()
        self.optimizer.step()

###################################################################################################

import torch.nn.functional as F
from torch import nn
import torch


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv(c_in, c_out, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False),
        norm(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )


class conv_transpose(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d):
        super(conv_transpose, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(c_out)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.conv_t(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class RefineModel(nn.Module):
    def __init__(self):
        super(RefineModel, self).__init__()
        self.conv1 = conv(6, 64)
        self.conv2 = conv(64, 128)
        self.conv3 = conv(128, 256)
        self.conv4 = conv(256, 512)
        self.conv5 = conv(512, 512)

        self.conv_t1 = conv_transpose(512, 512)
        self.conv_t2 = conv_transpose(1024, 256)
        self.conv_t3 = conv_transpose(512, 128)
        self.conv_t4 = conv_transpose(256, 64)

        self.conv6 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.apply(init_weights)

    def forward(self, dY_Yst):
        enc1 = self.conv1(dY_Yst)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        dec1 = self.conv5(enc4)
        dec2 = self.conv_t1(dec1, enc4)
        dec3 = self.conv_t2(dec2, enc3)
        dec4 = self.conv_t3(dec3, enc2)
        dec5 = self.conv_t4(dec4, enc1)

        y = F.interpolate(dec5, scale_factor=2, mode='bilinear', align_corners=True)
        y = self.conv6(y)

        return torch.tanh(y)
