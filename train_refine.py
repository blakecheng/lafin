import os
import torch
import torch.optim as optim

from src.dataset import Dataset
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from src.loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss
from src.models import InpaintGenerator, RefineModel, InpaintingModel

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



path = "/home/public/cb/code/lafin/checkpoints/obama-ae-lmin-256-512-latent-refine"
# path = "/home/public/cb/code/lafin/remote_checkpoint/celeba1024-all-Jul201101"
gen_weights_path = os.path.join(path,'InpaintingModel_gen.pth')
data = torch.load(gen_weights_path)

coarse_cfg = Config(os.path.join(path,"config.yml"))
Inpaintor = InpaintingModel(coarse_cfg).cuda()
generator = Inpaintor.generator
# generator = nn.DataParallel(generator)
generator.load_state_dict(data['generator'])
generator.eval()

refine_path = os.path.join(path,"refine_Obama")
create_dir(os.path.join(refine_path,"checkpoint"))
create_dir(os.path.join(refine_path,"sample"))
config_path = os.path.join(refine_path,"refine_config.yml")

config = Config(config_path)
print(config)
config.MODEL = 2
config.MODE = 2 





# generator = generator.cuda()

iteration = data['iteration']

refinetor = RefineModel()
refinetor = refinetor.cuda()

# re_data = torch.load(os.path.join(refine_path,'checkpoint/refine.pth'))
# refinetor.load_state_dict(re_data['refiner'])

train_dataset = Dataset(config,config.TRAIN_INPAINT_IMAGE_FLIST,config.TRAIN_INPAINT_LANDMARK_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=1,
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

l1_loss = nn.L1Loss().cuda()
perceptual_loss = PerceptualLoss().cuda()
style_loss = StyleLoss().cuda()
tv_loss = TVLoss().cuda()

refinetor.add_module('l1_loss', l1_loss)
refinetor.add_module('perceptual_loss', perceptual_loss)
refinetor.add_module('style_loss', style_loss)
refinetor.add_module('tv_loss', tv_loss)

ref_optimizer = optim.Adam(
            params=refinetor.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

psnr = PSNR(255.0).cuda()
cal_mae = nn.L1Loss(reduction='sum')

for epoch in range(10000):
    print("Epoch %d " % epoch)
    progbar = Progbar(len(train_loader), width=20, stateful_metrics=['epoch'])
    for items in train_loader:
        generator.eval()
        refinetor.train()
        
        ref_optimizer.zero_grad()
        
        images, landmarks, masks = cuda(*items)
        
        landmarks[landmarks >= config.INPUT_SIZE] = config.INPUT_SIZE - 1
        landmarks[landmarks < 0] = 0
        landmark_map = torch.zeros((landmarks.shape[0], 1, config.INPUT_SIZE, config.INPUT_SIZE)).cuda()
        for i in range(landmarks.shape[0]):
            landmark_map[i, 0, landmarks[i, 0:config.LANDMARK_POINTS, 1], landmarks[i,0:config.LANDMARK_POINTS,0]] = 1

        outputs = Inpaintor(images, landmark_map, masks)
        # landmarks = landmark_map
        # images_masked = (images * (1 - masks).float()) + masks
        # inputs = torch.cat((images_masked, landmarks), dim=1)
        # scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
        #                             mode='bilinear', align_corners=True)
        # scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
        #                             mode='bilinear', align_corners=True)
        
        # outputs = generator(inputs,masks,scaled_masks_half,scaled_masks_quarter)
        
        batch_size = images.shape[0]
        index = torch.randperm(batch_size).cuda()
        
        ref_images, ref_landmarks, ref_masks = images[index], landmarks[index], masks[index]
        ref_landmarks[ref_landmarks >= config.INPUT_SIZE] = config.INPUT_SIZE - 1
        ref_landmarks[ref_landmarks < 0] = 0
        ref_landmark_map = torch.zeros((ref_landmarks.shape[0], 1, config.INPUT_SIZE, config.INPUT_SIZE)).cuda()
        for i in range(ref_landmarks.shape[0]):
            ref_landmark_map[i, 0, ref_landmarks[i, 0:config.LANDMARK_POINTS, 1], ref_landmarks[i,0:config.LANDMARK_POINTS,0]] = 1

        ref_outputs = Inpaintor(ref_images, ref_landmark_map, ref_masks)
        
        coarse = torch.cat((ref_outputs-ref_images,outputs),dim=1) 
        refine_result = refinetor(coarse)
        
        ref_l1_loss = refinetor.l1_loss(refine_result, images) * config.L1_LOSS_WEIGHT / torch.mean(masks)
        ref_content_loss = refinetor.perceptual_loss(refine_result, images)
        ref_style_loss = refinetor.style_loss(refine_result * masks, images * masks)
        ref_tv_loss = refinetor.tv_loss(refine_result*masks+images*(1-masks))

        ref_loss = 0
        ref_loss += ref_l1_loss
        ref_loss += ref_content_loss * config.CONTENT_LOSS_WEIGHT
        ref_loss += ref_style_loss * config.STYLE_LOSS_WEIGHT
        ref_loss += ref_tv_loss * config.TV_LOSS_WEIGHT
        
        #outputs_merged = (outputs * masks) + (images * (1-masks))
        v_psnr = psnr(postprocess(images), postprocess(outputs))
        v_mae = (torch.sum(torch.abs(images - outputs)) / torch.sum(images)).float()
        
        logs = [
            ("ref_l1_loss",ref_l1_loss.item()),
            ("ref_content_loss",ref_content_loss.item()),
            ("ref_style_loss",ref_style_loss.item()),
            ("ref_tv_loss",ref_tv_loss.item()),
            ("v_psnr",v_psnr.item()),
            ("v_mae",v_mae.item())
        ]
        
        logs = [
                    ("epoch", epoch),
                ] + logs
        
        progbar.add(len(images), values=logs)
        write_log(os.path.join(refine_path,"log.dat"),logs)
        
        ref_loss.backward()
        ref_optimizer.step()
        
        sample_image = stitch_images(
            postprocess(images),
            postprocess(landmark_map),
            postprocess(outputs),
            postprocess(ref_images),
            postprocess(refine_result),
            img_per_row = 2
        )
        sample_image.save(os.path.join(refine_path,"sample",str(epoch).zfill(5)+".png"))
    
    torch.save({
        'iteration': epoch,
        'refiner': refinetor.state_dict()
    }, os.path.join(refine_path,'checkpoint/refine.pth'))

