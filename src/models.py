import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss,Landmark_loss,New_AdversarialLoss,DICELoss,FeaturematchingLoss
from .stylegan2 import stylegan_L2I_Generator,stylegan_L2I_Generator2,stylegan_L2I_Generator3,stylegan_L2I_Generator4,stylegan_L2I_Generator5,stylegan_L2I_Generator_AE
from .stylegan2 import stylegan_L2I_Generator_AE_landmark_in,stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in
from .stylegan2 import ref_guided_inpaintor,stylegan_ae_facereenactment,stylegan_base_facereenactment,stylegan_base_faceswap,stylegan_base_faceae
from .stylegan2 import stylegan_rotate,wide_stylegan_rotate
# from .stylegan2 import dualnet
from .res_unet import MultiScaleResUNet
from .faceshifter_generator import faceshifter_inpaintor,faceshifter_reenactment,faceshifter_reenactment2
from .oneshot_facereenactment import Normal_Encoder
from .stylegan2 import Discriminator as style_dis
from .patch_gan import Discriminator as patch_dis

# from .faceshifter_generator import faceshifter_sin
import numpy as np


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
        if hasattr(config, 'DISTRIBUTED') and config.DISTRIBUTED == True:
            self.local_rank = config.LocalRank
            
        
        self.is_local = False
        if hasattr(config, 'LOCAL'):
            self.is_local = config.LOCAL
            print("local mode：{}".format(config.LOCAL))
        # generator input: [rgb(3) + landmark(1)]
        # discriminator input: [rgb(3)]

        self.config = config
        
        self.inpaint_type = None
        if hasattr(config, 'INPAINTOR'):
            self.inpaint_type = config.INPAINTOR

        from .augment import AugWrapper

        self.augwrapper = AugWrapper(config.INPUT_SIZE).cuda()

        if self.inpaint_type == "face_rotate":
            print("#####################")
            print("Inpainting_for_face_rotate!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            arc_eval = config.ARC_EVAL
            is_transparent = True if (hasattr(config, 'LOSS_TYPE') and config.LOSS_TYPE=="learn_mask") else False
            num_init_filters = 4 if (hasattr(config, 'LM_TYPE') and config.LM_TYPE!="landmark") else 1
            generator = stylegan_rotate(image_size=image_size, \
                latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers, \
                num_init_filters = num_init_filters, arc_eval=arc_eval,transparent = is_transparent)
        elif self.inpaint_type == "face_rotate_wide":
            print("#####################")
            print("Inpainting_for_face_rotate!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            arc_eval = config.ARC_EVAL
            is_transparent = True if (hasattr(config, 'LOSS_TYPE') and config.LOSS_TYPE=="learn_mask") else False
            num_init_filters = 4 if (hasattr(config, 'LM_TYPE') and config.LM_TYPE!="landmark") else 1
            generator = wide_stylegan_rotate(image_size=image_size, \
                network_capacity=network_capacity,num_layers=num_layers, \
                num_init_filters = num_init_filters, arc_eval=arc_eval,transparent = is_transparent,fmap_max=1024)
        elif self.inpaint_type == "MSG_Inpainting_for_face_swap":
            print("#####################")
            print("MSG_Inpainting_for_face_swap, USE stylegan generator, AE landmark!")
            print("#####################\n")
            from src.msg_stylegan2 import msg_stylegan2_lm_id_G
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY

            generator = msg_stylegan2_lm_id_G(image_size=image_size, \
                latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers, \
                num_init_filters = 4)
        elif self.inpaint_type == "Inpainting_for_face_swap":
            print("#####################")
            print("Inpainting_for_face_swap, USE stylegan generator, AE landmark!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            arc_eval = config.ARC_EVAL
            generator = stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in(image_size=image_size, \
                latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers, \
                num_init_filters = 4)
        elif self.inpaint_type == "stylegan_base_faceae":
            print("#####################")
            print("USE stylegan_base_faceae generator!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            fmap_max =  config.FMAP_MAX
            print(image_size,fmap_max,latent_dim)
            generator = stylegan_base_faceae(image_size=image_size, fmap_max= fmap_max,latent_dim= latent_dim)
        elif self.inpaint_type == "stylegan_base_faceswap":
            print("#####################")
            print("USE stylegan_base_faceswap generator!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            fmap_max =  config.FMAP_MAX
            generator = stylegan_base_faceswap(image_size=image_size, fmap_max= fmap_max,latent_dim= latent_dim)
        elif self.inpaint_type == "stylegan_base_facereenactment":
            print("#####################")
            print("USE stylegan_base_facereenactment generator!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            fmap_max =  config.FMAP_MAX
            generator = stylegan_base_facereenactment(image_size=image_size, fmap_max= fmap_max,latent_dim= latent_dim)
        elif  self.inpaint_type == "stylegan2":
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
        elif self.inpaint_type == "s2_ae_landmark_and_arcfaceis_in":
            print("#####################")
            print("USE stylegan generator, AE landmark and arcfaceid!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            arc_eval = config.ARC_EVAL
            print(hasattr(config, 'LM_TYPE'), config.LM_TYPE!="landmark")
            num_init_filters = 4 if ((hasattr(config, 'LM_TYPE') and config.LM_TYPE!="landmark")) else 1
            print("num_init_filters is ", num_init_filters)
            generator = stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers, \
                num_init_filters=num_init_filters, arc_eval = arc_eval)
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
        elif self.inpaint_type == "faceshifter_inpaintor_selfref":
            print("#####################")
            print("USE faceshifter inpaintor!")
            print("#####################\n")
            generator = faceshifter_inpaintor()
        elif self.inpaint_type == "faceshifter_reenactment2":
            print("#####################")
            print("USE faceshifter inpaintor pairedref!")
            print("#####################\n")
            generator = faceshifter_reenactment2()
        elif self.inpaint_type == "faceshifter_reenactment":
            print("#####################")
            print("USE faceshifter inpaintor!")
            print("#####################\n")
            generator = faceshifter_reenactment()
        elif self.inpaint_type == "stylegan_ae_facereenactment":
            print("#####################")
            print("USE stylegan ae facereenactment !")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            generator = stylegan_ae_facereenactment(image_size=image_size,latent_dim=latent_dim)
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
        

        if (hasattr(config, 'LM_TYPE') and self.config.LM_TYPE!="landmark"):
            in_channels = 7
        else:
            in_channels = 4
        #discriminator = Discriminator(in_channels=4, use_sigmoid=config.GAN_LOSS != 'hinge')

        if self.inpaint_type == "MSG_Inpainting_for_face_swap":
            from src.msg_stylegan2 import msg_stylegan2_lm_id_D
            num_layers = config.NUM_LAYERS -1 
            discriminator = msg_stylegan2_lm_id_D(depth=num_layers)
        elif hasattr(config, 'DISCRIMINATOR') and config.DISCRIMINATOR == "no_landmark":
            from src.no_landmark import no_landmark_Discriminator
            discriminator = no_landmark_Discriminator(padding="zero",in_channels=3,out_channels=3,num_channels=64,
                          max_num_channels=512,embed_channels=512,dis_num_blocks=7,image_size=256,
                         num_labels=512)
        elif hasattr(config, 'DISCRIMINATOR') and config.DISCRIMINATOR == "lafin":
            discriminator = Discriminator(in_channels)
        elif hasattr(config, 'DISCRIMINATOR') and config.DISCRIMINATOR == "patch":
            discriminator = patch_dis(in_channels=in_channels)
        else:
            discriminator = style_dis(image_size = config.INPUT_SIZE,transparent=in_channels)
        

        if use_apex == True:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
        else:
            # if len(config.GPU) > 1:
            #     if hasattr(config, 'DISTRIBUTED') and config.DISTRIBUTED == True:
            #         generator = torch.nn.parallel.DistributedDataParallel(generator.cuda(),device_ids=[self.local_rank],output_device=self.local_rank,find_unused_parameters=True)
            #         discriminator = torch.nn.parallel.DistributedDataParallel(discriminator.cuda(),device_ids=[self.local_rank],output_device=self.local_rank,find_unused_parameters=True)
            #     else:
            #         generator = nn.DataParallel(generator)
            #         discriminator = nn.DataParallel(discriminator)
            # else:
            generator = generator.cuda()
            discriminator = discriminator.cuda()


        l1_loss = nn.L1Loss().cuda()
        perceptual_loss = PerceptualLoss(net='pytorch').cuda()
        # from lpips_pytorch import LPIPS
        # criterion = LPIPS(
        #         net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
        #         version='0.1'  # Currently, v0.1 is supported
        # )
        # perceptual_loss = criterion.cuda()

        
        style_loss = StyleLoss().cuda()

        
        

        if hasattr(self.config, 'LOSS_TYPE') and self.config.LOSS_TYPE == "learn_mask":
            adversarial_loss = New_AdversarialLoss(gan_type=config.GAN_LOSS).cuda()
            self.dice_loss = DICELoss()
        elif hasattr(self.config, 'LOSS_TYPE') and self.config.LOSS_TYPE == "latent_pose":
            adversarial_loss = New_AdversarialLoss(gan_type=config.GAN_LOSS).cuda()
            self.fm_loss = FeaturematchingLoss()
        else:
            adversarial_loss = AdversarialLoss(type=config.GAN_LOSS).cuda()

        self.tv_loss = TVLoss().cuda()
        
        

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        
        # if self.inpaint_type == "faceshifter_reenactment2" or self.inpaint_type == "stylegan_ae_facereenactment" \
        #     or self.inpaint_type == "stylegan_base_facereenactment" or self.inpaint_type == "stylegan_base_faceswap" \
        #     or self.inpaint_type == "stylegan_base_faceae"   :
        landmark_loss = Landmark_loss()
        self.add_module('landmark_loss',landmark_loss)
        id_loss = PerceptualLoss(net='face').cuda()
        self.add_module('id_loss',id_loss)
            
        
        # Apex

        
        
        self.gen_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad,generator.parameters()),
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

        # def get_loss1(self,images, landmarks, masks,outputs):
    #     gen_loss = 0
    #     dis_loss = 0


    #     # discriminator loss
    #     dis_input_real = images
    #     dis_input_fake = outputs.detach()
    #     if self.is_local == True:
    #         dis_real, _ = self.discriminator(torch.cat((dis_input_real* masks, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
    #         dis_fake, _ = self.discriminator(torch.cat((dis_input_fake* masks, landmarks), dim=1))  
    #     else:
    #         dis_real, _ = self.discriminator(torch.cat((dis_input_real, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
    #         dis_fake, _ = self.discriminator(torch.cat((dis_input_fake, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
    #     dis_real_loss = self.adversarial_loss(dis_real, True, True)
    #     dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
    #     dis_loss += (dis_real_loss + dis_fake_loss) / 2


    #     # generator adversarial loss
    #     gen_input_fake = outputs
    #     if self.is_local == True:
    #         gen_fake, _ = self.discriminator(torch.cat((gen_input_fake* masks, landmarks), dim=1))
    #     else:
    #         gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]
    #     gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
    #     gen_loss += gen_gan_loss

    #     # generator l1 loss
    #     if self.is_local == True:
    #         gen_l1_loss = self.l1_loss(outputs* masks, images* masks) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
    #     else:
    #         gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
    #     gen_loss += gen_l1_loss


    #     # generator perceptual loss
    #     gen_content_loss = self.perceptual_loss(outputs, images)
    #     gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
    #     gen_loss += gen_content_loss


    #     # generator style loss
    #     gen_style_loss = self.style_loss(outputs * masks, images * masks)
    #     gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
    #     gen_loss += gen_style_loss

    #     #generator tv loss
    #     tv_loss = self.tv_loss(outputs*masks+images*(1-masks))
    #     gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

    #     # create logs
    #     logs = [
    #         ("gLoss",gen_loss.item()),
    #         ("ggan_l",gen_gan_loss.item()),
    #         ("gl1_l",gen_l1_loss.item()),
    #         ("gcontent_l",gen_content_loss.item()),
    #         ("gstyle_l",gen_style_loss.item()),
    #         ("gtv_l",tv_loss.item()),
    #         ("dLoss",dis_loss.item())
    #     ]

    #     return outputs, gen_loss, dis_loss, logs

    
    def get_loss_latent(self,images,landmark,masks,outputs):
        gen_loss = 0
        dis_loss = 0


        fake_features,real_features,fake_score_G,fake_score_D,real_score = self.discriminator(outputs,images)

        gen_gan_loss, dis_loss = self.adversarial_loss(fake_score_G,fake_score_D,real_score)

        gen_gan_loss = gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        gen_id_loss = self.config.ID_LOSS_WEIGHT* self.id_loss(outputs, images*masks,None)
        gen_loss += gen_id_loss

        gen_content_loss = self.perceptual_loss(outputs, images*masks)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        gen_matching_loss = self.fm_loss(outputs,images*masks)
        gen_matching_loss = gen_matching_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_matching_loss

        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gen_id",gen_id_loss.item()),
            ("gen_fm",gen_matching_loss.item()),
            ("dLoss",dis_loss.item())
        ]
        return outputs, gen_loss, dis_loss, logs





    def get_loss_inpainting(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        # if self.is_local == True:
        #     dis_real, _ = self.discriminator(torch.cat((dis_input_real* masks, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        #     dis_fake, _ = self.discriminator(torch.cat((dis_input_fake* masks, landmarks), dim=1))  
        # else:
        dis_real, _ = self.discriminator(torch.cat((dis_input_real* masks, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
  
        gen_l1_loss = self.l1_loss(outputs, images*masks) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images*masks)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs, images*masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #generator tv loss
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

        # zid loss
        
        gen_id_loss = self.config.ID_LOSS_WEIGHT* self.id_loss(outputs, images*masks)
        gen_loss += gen_id_loss
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(outputs, images*masks) 
        gen_loss += gen_landmark_loss

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            ("gen_id",gen_id_loss.item()),
            ("gen_lm",gen_landmark_loss.item()),
            ("dLoss",dis_loss.item())
        ]
        return outputs, gen_loss, dis_loss, logs
    
    def get_loss_mask(self,images, landmarks, masks,outputs,landmarks_points):
        gen_loss = 0
        dis_loss = 0

        pred_mask = outputs[:, -1:]
        pred_images = outputs[:,:-1]
        outputs = pred_images*pred_mask

        # discriminator and generator loss
        dis_input_real = images*masks
        dis_input_fake = outputs.detach()
        gen_input_fake = outputs


        dis_real, _ = self.discriminator(torch.cat((dis_input_real, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake, landmarks), dim=1)) 
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                  # in: [rgb(3)+landmark(1)]
  
        gen_gan_loss, dis_loss = self.adversarial_loss(gen_fake,dis_fake,dis_real)

        gen_gan_loss = gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
  
        gen_l1_loss = self.l1_loss(outputs, images*masks) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images*masks)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs, images*masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        
        # generator dice loss
        gen_dice_loss = self.dice_loss(pred_mask,masks)
        gen_dice_loss = gen_dice_loss * self.config.DICE_LOSS_WEIGHT
        gen_loss += gen_dice_loss

        #generator tv loss
        # tv_loss = self.tv_loss(outputs)
        # gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

        # zid loss
        gen_id_loss = self.config.ID_LOSS_WEIGHT* self.id_loss(outputs, images*masks,None)
        gen_loss += gen_id_loss
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(outputs, images*masks) 
        gen_loss += gen_landmark_loss

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gdice_l",gen_dice_loss.item()),
            ("gen_id",gen_id_loss.item()),
            ("gen_lm",gen_landmark_loss.item()),
            ("dLoss",dis_loss.item())
        ]
        return outputs, gen_loss, dis_loss, logs


    def get_loss1(self,images, landmarks, masks,outputs):
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
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        else:
            gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT

        gen_loss += gen_gan_loss

        # generator l1 loss
        if self.is_local == True:
            gen_l1_loss = self.l1_loss(outputs* masks, images* masks) * self.config.L1_LOSS_WEIGHT
        else:
            gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        
        gen_loss += gen_l1_loss


        # generator perceptual loss
        if self.is_local == True:
            gen_content_loss = self.perceptual_loss(outputs* masks, images* masks)
        else:
            gen_content_loss = self.perceptual_loss(outputs, images)

        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += torch.mean(gen_content_loss)


        # generator style loss
        if self.is_local == True:
            gen_style_loss = self.style_loss(outputs* masks, images* masks)
        else:
            gen_style_loss = self.style_loss(outputs, images )
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #generator tv loss
        if self.is_local == True:
            tv_loss = self.tv_loss(outputs*masks)
        else:
            tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

        # zid loss
        gen_id_loss = self.config.ID_LOSS_WEIGHT* self.id_loss(outputs*masks, images*masks)
        gen_loss += gen_id_loss
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(outputs*masks, images*masks) 
        gen_loss += gen_landmark_loss

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            ("gen_id",gen_id_loss.item()),
            ("gen_lm",gen_landmark_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs
    
    def get_loss_msg(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        real_images = [images] + [torch.nn.functional.avg_pool2d(images, int(np.power(2, i)))
                                     for i in range(1, self.config.NUM_LAYERS-1)]
        real_images = list(reversed(real_images))

        outputs, multiscales = outputs

        dis_input_real = real_images
        dis_input_fake = list(map(lambda x: x.detach(), multiscales))
        dis_input_fake.pop(len(dis_input_fake)-2)
        # for real_I,fake_I in zip(dis_input_real,dis_input_fake):
        #     print(real_I.shape,fake_I.shape)

        dis_real = self.discriminator(dis_input_real)                   # in: [rgb(3)+landmark(1)]
        dis_fake = self.discriminator(dis_input_fake)                   # in: [rgb(3)+landmark(1)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = multiscales
        gen_input_fake.pop(len(gen_input_fake)-2)
        gen_fake  = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
  
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs, images )
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #generator tv loss
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

        # zid loss
        gen_id_loss = self.config.ID_LOSS_WEIGHT* self.id_loss(outputs*masks+images*(1-masks), images)
        gen_loss += gen_id_loss
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(outputs*masks+images*(1-masks), images) 
        gen_loss += gen_landmark_loss

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            ("gen_id",gen_id_loss.item()),
            ("gen_lm",gen_landmark_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs
    
    def get_loss_fr1(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0
        ref_landmarks,ref_images,outputs,z_id,zatt,is_same = outputs
        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        
        dis_real, _ = self.discriminator(torch.cat((dis_input_real , landmarks), dim=1))             
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake , landmarks), dim=1))  

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
      
        # generator adversarial loss
        
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]

        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss*is_same

        #generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss*is_same

        #generator style loss
        gen_style_loss = self.style_loss(outputs, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss*is_same

        #generator tv loss
        
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss
        
        # zid loss
        with torch.no_grad():
            Y_zatt = self.generator.module.get_zatt(outputs,landmarks)
        
        batch_size = images.shape[0]
        L_attr = 0
        for i in range(len(Y_zatt)):
            L_attr += torch.mean(torch.pow(Y_zatt[i] - zatt[i], 2).reshape(batch_size, -1), dim=1).mean()
        
        gen_att_loss = self.config.ATT_LOSS_WEIGHT * L_attr/ 2.0
        gen_loss += gen_att_loss
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(images,outputs) 
        gen_loss += gen_landmark_loss
        
        
        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            ("gatt_l",gen_att_loss.item()),
            ("glandmark_l",gen_landmark_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs
    
    
    
    def get_loss_fr2(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0
        ref_landmarks,ref_images,outputs,z_id,zatt,is_same = outputs
        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        
        dis_real, _ = self.discriminator(torch.cat((dis_input_real , landmarks), dim=1))             
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake , landmarks), dim=1))  

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2 *is_same
      
        # generator adversarial loss
        
        
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]

        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss*is_same

        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss*is_same

        #generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss*is_same

        #generator style loss
        gen_style_loss = self.style_loss(outputs, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss*is_same

        #generator tv loss
        
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss
        
        # zid loss
        with torch.no_grad():
            Y_zatt = self.generator.module.get_zatt(images,landmarks)
        
        batch_size = images.shape[0]
        L_attr = 0
        for i in range(len(Y_zatt)):
            L_attr += torch.mean(torch.pow(Y_zatt[i] - zatt[i], 2).reshape(batch_size, -1), dim=1).mean()
        
        gen_att_loss = self.config.ATT_LOSS_WEIGHT * L_attr/ 2.0
        gen_loss += gen_att_loss
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(images,outputs) 
        gen_loss += gen_landmark_loss
        
        
        # create logs
        if is_same==1:
            logs = [
                ("is_same",1),
                ("gLoss",gen_loss.item()),
                ("ggan_l",gen_gan_loss.item()),
                ("gl1_l",gen_l1_loss.item()),
                ("gcontent_l",gen_content_loss.item()),
                ("gstyle_l",gen_style_loss.item()),
                ("gtv_l",tv_loss.item()),
                ("gatt_l",gen_att_loss.item()),
                ("glandmark_l",gen_landmark_loss.item()),
                ("dLoss",dis_loss.item())
            ]
        else:
            logs = [
                ("is_same",0),
                ("gLoss",gen_loss.item()),
                ("ggan_l",gen_gan_loss.item()),
                ("gtv_l",tv_loss.item()),
                ("gatt_l",gen_att_loss.item()),
                ("glandmark_l",gen_landmark_loss.item()),
                ("dLoss",dis_loss.item())
            ]

        return outputs, gen_loss, dis_loss, logs
    
    def get_loss_stylegan1(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0
        ref_landmarks,ref_images,outputs,input_noise,styles,is_same = outputs
        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        
        dis_real, _ = self.discriminator(torch.cat((dis_input_real , landmarks), dim=1))             
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake , landmarks), dim=1))  

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2 
      
        # generator adversarial loss
        
        
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]

        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss 

        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss*is_same

        #generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss*is_same

        #generator style loss
        gen_style_loss = self.style_loss(outputs, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss*is_same

        #generator tv loss
        
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss
        
        # zatt loss
        with torch.no_grad():
            Y_zatt = self.generator.module.get_zatt(outputs,landmarks)
        
        batch_size = images.shape[0]
        L_attr = 0
 
        L_attr += torch.mean(torch.pow(Y_zatt - input_noise, 2).reshape(batch_size, -1), dim=1).mean()

        gen_att_loss = self.config.ATT_LOSS_WEIGHT * L_attr/ 2.0
        gen_loss += gen_att_loss
        
        # zid loss
        with torch.no_grad():
            Y_id = self.generator.module.get_style(outputs)
            L_id = self.config.ID_LOSS_WEIGHT* (1 - torch.cosine_similarity(styles, Y_id, dim=1)).mean() 
        gen_loss += L_id
        
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(images,outputs) 
        gen_loss += gen_landmark_loss
        
        
        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            ("gatt_l",gen_att_loss.item()),
            ("glandmark_l",gen_landmark_loss.item()),
            ("gid_l",L_id.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs
    
    
    def get_loss_stylegan_fr(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0
        images,landmarks,ref_images,ref_landmarks,outputs,rgbs,iatts,id_latent,lm_latent,is_same = outputs
        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        
        dis_real, _ = self.discriminator(torch.cat((dis_input_real , landmarks), dim=1))             
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake , landmarks), dim=1))  

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2 
      
        # generator adversarial loss
        
        
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]

        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss*is_same

        #generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss*is_same

        #generator style loss
        gen_style_loss = self.style_loss(outputs, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss*is_same

        #generator tv loss
        
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss
        
        # zatt loss
        # Y VS ref
        Y_zatts = self.generator.get_att(outputs,landmarks)
        
        batch_size = images.shape[0]
        L_attr = 0
        for i in range(len(iatts)):
            L_attr += torch.mean(torch.pow(Y_zatts[i] - iatts[i], 2).reshape(batch_size, -1), dim=1).mean()

        gen_att_loss = self.config.ATT_LOSS_WEIGHT * L_attr/ 2.0
        gen_loss += gen_att_loss
        
        # zid loss
        # Y VS ref
        with torch.no_grad():
            Y_id = self.generator.get_id_latent(outputs)
            L_id = self.config.ID_LOSS_WEIGHT* (1 - torch.cosine_similarity(id_latent, Y_id, dim=1)).mean() 
        gen_loss += L_id
        
        
        # landmark_loss
        # Y vs img
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(images,outputs) 
        gen_loss += gen_landmark_loss
        
        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            ("gatt_l",gen_att_loss.item()),
            ("glandmark_l",gen_landmark_loss.item()),
            ("gid_l",L_id.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs

    
    def get_loss_stylegan_fs(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0
        images,landmarks,ref_images,ref_landmarks,outputs,rgbs,iatts,id_latent,lm_latent,is_same = outputs
        # discriminator loss
        dis_input_real = ref_images
        dis_input_fake = outputs.detach()
        
        dis_real, _ = self.discriminator(torch.cat((dis_input_real , ref_landmarks), dim=1))             
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake , landmarks), dim=1))  

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2 
      
        # generator adversarial loss
        
        
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]

        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss*is_same

        #generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss*is_same

        #generator style loss
        gen_style_loss = self.style_loss(outputs, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss*is_same

        #generator tv loss
        
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss
        
        # zatt loss
        Y_zatts = self.generator.module.get_att(outputs,landmarks)
        
        batch_size = images.shape[0]
        L_attr = 0
        for i in range(len(iatts)):
            L_attr += torch.mean(torch.pow(Y_zatts[i] - iatts[i], 2).reshape(batch_size, -1), dim=1).mean()

        gen_att_loss = self.config.ATT_LOSS_WEIGHT * L_attr/ 2.0
        gen_loss += gen_att_loss
        
        # zid loss
        with torch.no_grad():
            Y_id = self.generator.module.get_id_latent(outputs)
            L_id = self.config.ID_LOSS_WEIGHT* (1 - torch.cosine_similarity(id_latent, Y_id, dim=1)).mean() 
        gen_loss += L_id
        
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(images,outputs) 
        gen_loss += gen_landmark_loss
        
        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            ("gatt_l",gen_att_loss.item()),
            ("glandmark_l",gen_landmark_loss.item()),
            ("gid_l",L_id.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs
    
    def get_loss_stylegan_ae(self,images, landmarks, masks,outputs):
        gen_loss = 0
        dis_loss = 0
        images,landmarks,outputs,rgbs,iatts,id_latent,lm_latent = outputs
        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        
        dis_real, _ = self.discriminator(torch.cat((dis_input_real , landmarks), dim=1))             
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake , landmarks), dim=1))  

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2 
      
        # generator adversarial loss
        
        
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]

        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss

        #generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        #generator style loss
        gen_style_loss = self.style_loss(outputs, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #generator tv loss
        
        tv_loss = self.tv_loss(outputs)
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss
        
        # zatt loss
        # Y_zatts = self.generator.get_att(outputs,landmarks)
        
        # batch_size = images.shape[0]
        # L_attr = 0
        # for i in range(len(iatts)):
        #     L_attr += torch.mean(torch.pow(Y_zatts[i] - iatts[i], 2).reshape(batch_size, -1), dim=1).mean()

        # gen_att_loss = self.config.ATT_LOSS_WEIGHT * L_attr/ 2.0
        # gen_loss += gen_att_loss
        
        # zid loss
        with torch.no_grad():
            Y_id = self.generator.get_id_latent(outputs)
            L_id = self.config.ID_LOSS_WEIGHT* (1 - torch.cosine_similarity(id_latent, Y_id, dim=1)).mean() 
        gen_loss += L_id
        
        # landmark_loss
        gen_landmark_loss = self.config.LM_LOSS_WEIGHT * self.landmark_loss(images,outputs) 
        gen_loss += gen_landmark_loss
        
        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("ggan_l",gen_gan_loss.item()),
            ("gl1_l",gen_l1_loss.item()),
            ("gcontent_l",gen_content_loss.item()),
            ("gstyle_l",gen_style_loss.item()),
            ("gtv_l",tv_loss.item()),
            #("gatt_l",gen_att_loss.item()),
            ("glandmark_l",gen_landmark_loss.item()),
            ("gid_l",L_id.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs
    
    
    
##################################################################################
    def process(self, images, landmarks, masks,landmarks_points=None):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, landmarks, masks)
        
        if self.inpaint_type == "faceshifter_reenactment":
            outputs, gen_loss, dis_loss, logs = self.get_loss_fr1(images, landmarks, masks,outputs)
        elif self.inpaint_type == "faceshifter_reenactment2":
            outputs, gen_loss, dis_loss, logs = self.get_loss_fr2(images, landmarks, masks,outputs)
        elif self.inpaint_type == "stylegan_ae_facereenactment":
            outputs, gen_loss, dis_loss, logs = self.get_loss_stylegan1(images, landmarks, masks,outputs)
        elif self.inpaint_type == "stylegan_base_facereenactment":
            outputs, gen_loss, dis_loss, logs = self.get_loss_stylegan_fr(images, landmarks, masks,outputs)
        elif self.inpaint_type == "stylegan_base_faceswap":
            outputs, gen_loss, dis_loss, logs = self.get_loss_stylegan_fs(images, landmarks, masks,outputs)
        elif self.inpaint_type == "stylegan_base_faceae":
            outputs, gen_loss, dis_loss, logs = self.get_loss_stylegan_ae(images, landmarks, masks,outputs)
        elif self.inpaint_type == "MSG_Inpainting_for_face_swap":
            outputs, gen_loss, dis_loss, logs = self.get_loss_msg(images, landmarks, masks,outputs)
        else:
            if hasattr(self.config, 'LOSS_TYPE') and self.config.LOSS_TYPE == "latent_pose":
                outputs, gen_loss, dis_loss, logs = self.get_loss_latent(images, landmarks, masks,outputs)
            if hasattr(self.config, 'LOSS_TYPE') and self.config.LOSS_TYPE == "inpainting":
                outputs, gen_loss, dis_loss, logs = self.get_loss_inpainting(images, landmarks, masks,outputs)
            elif hasattr(self.config, 'LOSS_TYPE') and self.config.LOSS_TYPE == "learn_mask":
                outputs, gen_loss, dis_loss, logs = self.get_loss_mask(images, landmarks, masks,outputs,landmarks_points)   
            else:
                outputs, gen_loss, dis_loss, logs = self.get_loss1(images, landmarks, masks,outputs)
    
        return outputs, gen_loss, dis_loss, logs
        

    def forward(self, images, landmarks, masks, id_images=None,Interpolation=False,alpha=0):
        if self.inpaint_type == "Inpainting_for_face_swap" or self.inpaint_type == "MSG_Inpainting_for_face_swap":
            batch_size = images.shape[0]
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks), dim=1)
            if id_images is None:
                id_images = images
            id_images = self.augwrapper(id_images)
            outputs = self.generator(inputs,id_images,input_noise=None,Interpolation=Interpolation,alpha=alpha)
        elif self.inpaint_type == "stylegan_base_faceae":
            batch_size = images.shape[0]
            output,rgbs,iatts,id_latent,lm_latent = self.generator(images,landmarks)
            return images,landmarks,output,rgbs,iatts,id_latent,lm_latent 
        elif self.inpaint_type == "stylegan_base_faceswap":
            batch_size = images.shape[0]
            if np.random.rand()> 0.8:
                is_same = 1
                ref_landmarks = landmarks
                ref_images = images
            else:
                is_same = 0
                ref_index = (torch.arange(batch_size)+1)%batch_size
                ref_landmarks = torch.clone(landmarks[ref_index])
                ref_images = torch.clone(images[ref_index])
            output,rgbs,iatts,id_latent,lm_latent = self.generator(images,landmarks,ref_images)
            return images,landmarks,ref_images,ref_landmarks,output,rgbs,iatts,id_latent,lm_latent,is_same 
        elif self.inpaint_type == "stylegan_base_facereenactment":
            batch_size = images.shape[0]
            if np.random.rand()> 0.8:
                is_same = 1
                ref_landmarks = landmarks
                ref_images = images
            else:
                is_same = 0
                ref_index = (torch.arange(batch_size)+1)%batch_size
                ref_landmarks = torch.clone(landmarks[ref_index])
                ref_images = torch.clone(images[ref_index])
            output,rgbs,iatts,id_latent,lm_latent = self.generator(landmarks,ref_images,ref_landmarks)
            return images,landmarks,ref_images,ref_landmarks,output,rgbs,iatts,id_latent,lm_latent,is_same 
        elif self.inpaint_type == "stylegan_ae_facereenactment":
            batch_size = images.shape[0]
            # ref_index = torch.randperm(batch_size).cuda()
            if np.random.rand()> 0.2:
                is_same = 1
                ref_landmarks = landmarks
                ref_images = images
            else:
                is_same = 0
                ref_index = (torch.arange(batch_size)+1)%batch_size
                ref_landmarks = torch.clone(landmarks[ref_index])
                ref_images = torch.clone(images[ref_index])
            rgb,input_noise,style = self.generator(landmarks,ref_images,ref_landmarks)
            return ref_landmarks,ref_images,rgb,input_noise,style,is_same
        elif self.inpaint_type == "s2_ae_landmark_in" :
            outputs = self.generator(landmarks)
        elif self.inpaint_type == "s2_ae_landmark_and_arcfaceis_in" or self.inpaint_type == "face_rotate" or self.inpaint_type == "face_rotate_wide":
            images = self.augwrapper(images)
            outputs = self.generator(landmarks,images)
        elif self.inpaint_type == "faceshifter":
            outputs = self.generator(images,landmarks,masks)
        elif self.inpaint_type == "faceshifter_inpaintor_selfref":
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks,masks), dim=1)
            ref_images = flip(images,dim=1)
            outputs,z_id,out_id = self.generator(inputs,ref_images)
        elif self.inpaint_type == "faceshifter_reenactment2":
            batch_size = images.shape[0]
            # ref_index = torch.randperm(batch_size).cuda()
            if np.random.rand()> 0.7:
                is_same = 1
                ref_landmarks = landmarks
                ref_images = images
            else:
                is_same = 0
                ref_index = (torch.arange(batch_size)+1)%batch_size
                ref_landmarks = torch.clone(landmarks[ref_index])
                ref_images = torch.clone(images[ref_index])
            outputs,z_id,out_id = self.generator(landmarks,ref_images,ref_landmarks)
            return ref_landmarks,ref_images,outputs,z_id,out_id,is_same
        elif self.inpaint_type == "faceshifter_reenactment":
            batch_size = images.shape[0]
            is_the_same = (torch.rand(batch_size)< 0.2).long().cuda()
            img_index = torch.arange(batch_size).cuda()
            ref_index = img_index*is_the_same.long() + ((img_index+1)%batch_size)*(1-is_the_same).long()
            ref_landmarks = landmarks[ref_index]
            ref_images = images[ref_index]
            outputs,z_id,zatt = self.generator(landmarks,ref_images,ref_landmarks)
            return ref_images,ref_landmarks,outputs,z_id,zatt,is_the_same
        elif "ref_guided" in self.inpaint_type:
            outputs = self.generator(images,landmarks,masks)
        elif "stylegan2" in self.inpaint_type:
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks), dim=1)
            if self.generator.training:
                outputs = self.generator(inputs)
            else:
                batch_size = images.shape[0]
                noise = torch.zeros(batch_size, images.shape[2], images.shape[3], 1).float().cuda()
                outputs = self.generator(inputs,noise)
        else:
            
            images_masked = self.augwrapper(images)
            inputs = torch.cat((images_masked, landmarks), dim=1)
            scaled_masks_quarter = F.interpolate(landmarks, size=[int(landmarks.shape[2] / 4), int(landmarks.shape[3] / 4)],
                                        mode='bilinear', align_corners=True)
            scaled_masks_half = F.interpolate(landmarks, size=[int(landmarks.shape[2] / 2), int(landmarks.shape[3] / 2)],
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
        
        
    def backward_fintune(self, gen_loss = None, dis_loss = None, update_mode=None):
        # Apex
        
        if update_mode == "freezeD":
            if use_apex == True:
                with amp.scale_loss(gen_loss, self.gen_optimizer) as scaled_loss: scaled_loss.backward()
            else:
                gen_loss.backward(retain_graph=True)
            self.gen_optimizer.step()
        else:
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

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

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

class landmark_loss(nn.Module):
    def __init__(self,points_num):
        super(landmark_loss,self).__init__()
        self.points_num = points_num
        lm_detector = MobileNetV2(points_num=points_num)
        lm_weight = torch.load("saved_models/landmark_detector.pth")
        lm_detector.load_state_dict(lm_weight['detector'])
        
    def forward(self,landmark_true, landmark_pred):
        landmark_loss = torch.norm((landmark_true-landmark_pred).reshape(-1,self.points_num*2),2,dim=1,keepdim=True)
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
