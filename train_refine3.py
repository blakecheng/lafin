import os
import torch
import torch.optim as optim

from src.dataset import Dataset
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from src.loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss, Landmark_loss,IDLoss
from src.models import InpaintGenerator, RefineModel, InpaintingModel
from src.networks import Discriminator

from src.config import Config
from tqdm import tqdm

from src.utils import Progbar, create_dir, stitch_images, imsave
from src.metrics import PSNR

from src.stylegan2 import stylegan_L2I_Generator,stylegan_L2I_Generator2,stylegan_L2I_Generator3,stylegan_L2I_Generator4,stylegan_L2I_Generator5,stylegan_L2I_Generator_AE
from src.stylegan2 import stylegan_L2I_Generator_AE_landmark_in,stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in
from src.stylegan2 import ref_guided_inpaintor,stylegan_ae_facereenactment
# from .stylegan2 import dualnet
from src.res_unet import MultiScaleResUNet
from src.faceshifter_generator import faceshifter_inpaintor,faceshifter_reenactment,faceshifter_reenactment2
from src.utils import Progbar, create_dir, stitch_images, imsave

import numpy as np

def cuda(*args):
    return (item.cuda() for item in args)

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

def write_log(path,logs):
    with open(path, 'a') as f:
        f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
        
def postprocess(img):
        # [0, 1] => [0, 255]
    img = img * 255.0
    img = torch.clamp(img,0,255)
    img = img.permute(0, 2, 3, 1)
    return img.int()



path = "/home/public/cb/code/lafin/checkpoints/celebahq-ae-lmfaceidin-256-512-latent_refine3"
# path = "/home/public/cb/code/lafin/remote_checkpoint/celeba1024-all-Jul201101"
gen_weights_path = os.path.join(path,'InpaintingModel_gen.pth')
data = torch.load(gen_weights_path,map_location="cuda:%s"%torch.cuda.current_device())

refine_path = os.path.join(path,"refine_celeba")
create_dir(os.path.join(refine_path,"checkpoint"))
create_dir(os.path.join(refine_path,"sample"))
config_path = os.path.join(refine_path,"refine_config.yml")

config = Config(config_path)


config.MODEL = 2
config.MODE = 2 


# coarse_cfg = Config(os.path.join(refine_path,"refine_config.yml"))
Inpaintor = InpaintingModel(config).cuda()
generator = Inpaintor.generator
# generator = nn.DataParallel(generator)
generator.load_state_dict(data['generator'])
generator.eval()


# generator = generator.cuda()

iteration = data['iteration']

image_size = config.INPUT_SIZE
latent_dim = 512
num_layers = 4
network_capacity = 64
refinetor = stylegan_L2I_Generator_AE(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers,in_c = 7)
refinetor = refinetor.cuda()

if os.path.exists(os.path.join(refine_path,'checkpoint/refine.pth')):
    re_data = torch.load(os.path.join(refine_path,'checkpoint/refine.pth'))
    start_iteration = re_data['iteration']
    refinetor.load_state_dict(re_data['refiner'])
  
print("*********create discriminator*****************") 
discriminator = Discriminator(in_channels=4, use_sigmoid=config.GAN_LOSS != 'hinge')
discriminator = discriminator.cuda()

if os.path.exists(os.path.join(refine_path,'checkpoint/refine_dis.pth')):
    re_data = torch.load(os.path.join(refine_path,'checkpoint/refine_dis.pth'))
    discriminator.load_state_dict(re_data['discriminator'])

print("*********create dataset*****************") 
train_dataset = Dataset(config,config.TRAIN_INPAINT_IMAGE_FLIST,config.TRAIN_INPAINT_LANDMARK_FLIST, config.TRAIN_MASK_FLIST, root=config.DATA_ROOT, augment=True, training=True)

len(train_dataset)

train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=2,
            drop_last=True,
            shuffle=True
        )


# val_dataset = Dataset(config,config.VAL_INPAINT_IMAGE_FLIST,config.VAL_INPAINT_LANDMARK_FLIST, config.VAL_MASK_FLIST, augment=True, training=True)
# train_loader = DataLoader(
#             dataset=val_dataset,
#             batch_size=config.BATCH_SIZE,
#             num_workers=1,
#             drop_last=True,
#             shuffle=True
#         )
# sample_iterator = iter(train_loader)

adversarial_loss = AdversarialLoss().cuda()
l1_loss = nn.L1Loss().cuda()
perceptual_loss = PerceptualLoss().cuda()
style_loss = StyleLoss().cuda()
tv_loss = TVLoss().cuda()
landmark_loss = Landmark_loss()
id_loss = IDLoss()

# refinetor.add_module('l1_loss', l1_loss)
# refinetor.add_module('perceptual_loss', perceptual_loss)
# refinetor.add_module('style_loss', style_loss)
# refinetor.add_module('tv_loss', tv_loss)
# refinetor.add_module('landmark_loss',landmark_loss)
# refinetor.add_module('id_loss',id_loss)

ref_optimizer = optim.Adam(
            params=refinetor.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

psnr = PSNR(255.0).cuda()
cal_mae = nn.L1Loss(reduction='sum')



for epoch in range(start_iteration,10000):
    print("Epoch %d " % epoch)
    progbar = Progbar(len(train_dataset), width=20, stateful_metrics=['epoch'])
    for items in train_loader:
        generator.eval()
        refinetor.train()
        discriminator.train()
        
        dis_optimizer.zero_grad()
        ref_optimizer.zero_grad()
        
        
        images, landmarks, masks = cuda(*items)
        
        
        landmarks[landmarks >= config.INPUT_SIZE] = config.INPUT_SIZE - 1
        landmarks[landmarks < 0] = 0
        landmark_map = torch.zeros((landmarks.shape[0], 1, config.INPUT_SIZE, config.INPUT_SIZE)).cuda()
        for i in range(landmarks.shape[0]):
            landmark_map[i, 0, landmarks[i, 0:config.LANDMARK_POINTS, 1], landmarks[i,0:config.LANDMARK_POINTS,0]] = 1
            
        batch_size = images.shape[0]
        if np.random.rand()> 0.8:
            is_same = 1
            ref_landmark_map = landmark_map
            ref_images = images
            ref_masks = masks
        else:
            is_same = 0
            ref_index = (torch.arange(batch_size)+1)%batch_size
            ref_landmark_map = torch.clone(landmark_map[ref_index])
            ref_images = torch.clone(images[ref_index])
            ref_masks = torch.clone(masks[ref_index])
        
        # index = torch.randperm(batch_size).cuda()
        # ref_images, ref_landmark_map, ref_masks = images[index], landmark_map[index], masks[index]

        outputs = Inpaintor(ref_images, landmark_map, ref_masks)

        ref_outputs = Inpaintor(ref_images, ref_landmark_map, ref_masks)
        
        coarse = torch.cat((ref_outputs-ref_images,outputs,landmark_map),dim=1) 
        
        refine_result = refinetor(coarse)
        
        # discriminator loss
        
        dis_loss = 0
        dis_input_real = images
        dis_input_fake = refine_result.detach()
 
        dis_real, _ = discriminator(torch.cat((dis_input_real, landmark_map), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_fake, _ = discriminator(torch.cat((dis_input_fake, landmark_map), dim=1))   

        dis_real_loss = adversarial_loss(dis_real, True, True)
        dis_fake_loss = adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        
        dis_loss.backward(retain_graph=True)
        dis_optimizer.step()
   
        
        
        # generator adversarial loss
        gen_input_fake = refine_result
        gen_fake, _ = discriminator(torch.cat((gen_input_fake, landmark_map), dim=1))
        ref_gan_loss = adversarial_loss(gen_fake, True, False) * config.INPAINT_ADV_LOSS_WEIGHT
        

        
        ref_l1_loss = l1_loss(refine_result, images) * config.L1_LOSS_WEIGHT *is_same
        ref_content_loss = perceptual_loss(refine_result, images)* config.CONTENT_LOSS_WEIGHT*is_same
        ref_style_loss = style_loss(refine_result, images)* config.STYLE_LOSS_WEIGHT*is_same
        ref_tv_loss = tv_loss(refine_result) * config.TV_LOSS_WEIGHT
        
        ref_lm_loss = landmark_loss(refine_result,images)*config.LM_LOSS_WEIGHT
        ref_id_loss = id_loss(refine_result,ref_images)*config.ID_LOSS_WEIGHT

        ref_loss = 0
        ref_loss += ref_gan_loss
        ref_loss += ref_l1_loss
        ref_loss += ref_content_loss 
        ref_loss += ref_style_loss 
        ref_loss += ref_tv_loss
        ref_loss += ref_lm_loss
        ref_loss += ref_id_loss
        
        #outputs_merged = (outputs * masks) + (images * (1-masks))
        v_psnr = psnr(postprocess(images), postprocess(outputs))
        v_mae = (torch.sum(torch.abs(images - outputs)) / torch.sum(images)).float()
        
        logs = [
            ("ref_gan_loss",ref_gan_loss.item()),
            ("ref_l1_loss",ref_l1_loss.item()),
            ("ref_content_loss",ref_content_loss.item()),
            ("ref_style_loss",ref_style_loss.item()),
            ("ref_tv_loss",ref_tv_loss.item()),
            ("ref_lm_loss",ref_lm_loss.item()),
            ("ref_id_loss",ref_id_loss.item()),
            ("ref_loss",ref_id_loss.item()),
            ("dis_loss",dis_loss.item()),
            ("v_psnr",v_psnr.item()),
            ("v_mae",v_mae.item())
        ]
        
        logs = [
                    ("epoch", epoch),
                ] + logs
        
        progbar.add(len(images), values=logs)
        write_log(os.path.join(refine_path,"log.dat"),logs)
        
        
        
        
        ref_loss.backward(retain_graph=True)
        ref_optimizer.step()
        
        
        sample_image = stitch_images(
            postprocess(images),
            postprocess(landmark_map),
            postprocess(outputs),
            postprocess(ref_images),
            postprocess(refine_result),
            img_per_row = 1
        )
        #print(images.shape,landmark_map.shape,outputs.shape,ref_images.shape,ref_images.shape,refine_result.shape)
        sample_image.save(os.path.join(refine_path,"sample",str(epoch).zfill(5)+".png"))
    
        if epoch%500==0:
            torch.save({
                'iteration': epoch,
                'refiner': refinetor.state_dict()
            }, os.path.join(refine_path,'checkpoint/refine.pth'))
            
            torch.save({
                'iteration': epoch,
                'discriminator': discriminator.state_dict()
            }, os.path.join(refine_path,'checkpoint/refine_dis.pth'))

