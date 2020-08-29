import os
os.chdir("/home/public/cb/code/lafin")

from src.config import Config
from src.networks import InpaintGenerator
import os
import torch
from src.stylegan2 import stylegan_L2I_Generator,stylegan_L2I_Generator2,stylegan_L2I_Generator3,stylegan_L2I_Generator4,stylegan_L2I_Generator5,stylegan_L2I_Generator_AE
from src.stylegan2 import ref_guided_inpaintor,stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in
from src.res_unet import MultiScaleResUNet
import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np  
import cv2  
import face_alignment
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class inpainting_api():
    def __init__(self,checkpoint):
        config_path = os.path.join(checkpoint,"config.yml")
        weight_path = os.path.join(checkpoint,"InpaintingModel_gen.pth")
        config = Config(config_path)

        self.inpaint_type = "lafin_origin"
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
        elif self.inpaint_type == "s2_ae_landmark_and_arcfaceis_in":
            print("#####################")
            print("USE stylegan generator, AE landmark and arcfaceid!")
            print("#####################\n")
            image_size = config.INPUT_SIZE
            latent_dim = config.LATENT
            num_layers = config.NUM_LAYERS
            network_capacity = config.NETWORK_CAPACITY
            generator = stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
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

        self.INPUT_SIZE = config.INPUT_SIZE
        
        data = torch.load(weight_path)
        generator.load_state_dict(data["generator"])
        generator.eval()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        self.generator = generator.cuda()
        
    def load_checkpoint(self,weight_path):
        data = torch.load(weight_path)
        self.generator.load_state_dict(data["generator"])
        

    def gen_result(self,images, landmarks, masks):
        if "stylegan2" in self.inpaint_type:
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks), dim=1)
            outputs = self.generator(inputs)
        elif self.inpaint_type == "s2_ae_landmark_in" :
            outputs = self.generator(landmarks)
        elif self.inpaint_type == "s2_ae_landmark_and_arcfaceis_in":
            outputs = self.generator(landmarks,images)
        elif self.inpaint_type == "faceshifter" :
            outputs = self.generator(images,landmarks,masks)
        elif self.inpaint_type == "faceshifter_inpaintor_selfref":
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks,masks), dim=1)
            ref_images = flip(images,dim=1)
            outputs,z_id,out_id = self.generator(inputs,ref_images)
        elif self.inpaint_type == "faceshifter_reenactment2":
            batch_size = images.shape[0]
            # ref_index = torch.randperm(batch_size).cuda()
            ref_index = (torch.arange(batch_size)+1)%batch_size
            ref_landmarks = landmarks[ref_index]
            ref_images = images[ref_index] 
            outputs,z_id,out_id = self.generator(landmarks,ref_images,ref_landmarks)
            return ref_landmarks,ref_images,outputs,z_id,out_id
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
        else:
            images_masked = (images * (1 - masks).float()) + masks
            inputs = torch.cat((images_masked, landmarks), dim=1)
            scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                        mode='bilinear', align_corners=True)
            scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                        mode='bilinear', align_corners=True)
            outputs = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter) 
        
        return outputs

    def resize(self, img, mask,landmarks, height, width):
        size_before = img.shape
        img = scipy.misc.imresize(img, [height, width])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        img_t = torchvision.transforms.functional.to_tensor(img).float()
        
        mask = scipy.misc.imresize(mask, [height, width])
        mask = rgb2gray(mask)
        mask = (mask > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask)
        mask_t = torchvision.transforms.functional.to_tensor(mask).float()
        
        side = np.minimum(size_before[0],size_before[1])
        
        landmarks[:,0] *= (height/side)
        landmarks[:,1] *= (width/side)
        landmarks = (landmarks+0.5).astype(np.int16)
        landmarks = torch.from_numpy(landmarks).long()
        
        return img_t,mask_t,landmarks
    

    def get_input(self,path,use_crop=False):

        INPUT_SIZE = self.INPUT_SIZE

        input_img = cv2.imread(path)
        if use_crop:
            preds = self.fa.get_landmarks(input_img)  
            if len(preds[0])==68:
                ## mask
                x, y, w, h = cv2.boundingRect(np.array(preds[0]))
                l = int(max(w,h))
                x = int(x-(l-w)/2)
                y = int(y-(l-h)*1.5/2)
                face = input_img.copy()[y:y+l,x:x+l]
                print("Crop image!")
                self.SHOW([self.to_plt(input_img),self.to_plt(face)])
        else:
            face = input_img
        
        preds = self.fa.get_landmarks(face)  
        x, y, w, h = cv2.boundingRect(np.array(preds[0][3:14]))
        mask_img = cv2.rectangle(face.copy(), (x, y), (x + w, y + h), (255, 255, 255), -1)
        raw_mask = cv2.rectangle(np.zeros(face.shape,face.dtype), (x, y), (x + w, y + h), (255, 255, 255), -1)

        landmark_cord = preds[0]
        landmark_img = np.zeros(face.shape,face.dtype)
        for i in range(landmark_cord.shape[0]):
            center = (int(landmark_cord[i,0]),int(landmark_cord[i,1]))
            cv2.circle(landmark_img,center, radius=5,color=(255, 255, 255), thickness=-1)
        
        images,masks,landmarks = self.resize(face,raw_mask,landmark_cord,INPUT_SIZE,INPUT_SIZE)

        landmarks[landmarks >= INPUT_SIZE-1] = INPUT_SIZE-1
        landmarks[landmarks < 0] = 0

        landmark_map = torch.zeros(1, INPUT_SIZE,INPUT_SIZE)
        landmark_map[0, landmarks[0:INPUT_SIZE, 1], landmarks[0:INPUT_SIZE, 0]] = 1
        images = images.unsqueeze(dim=0)
        masks = masks.unsqueeze(dim=0)
        landmark_map = landmark_map.unsqueeze(dim=0).float()

        return images,landmark_map,masks, face,landmark_img, raw_mask
    
    def get_landmark(self,input_img,use_crop=False):
        INPUT_SIZE = self.INPUT_SIZE
        if use_crop:
            preds = self.fa.get_landmarks(input_img)  
            if len(preds[0])==68:
                ## mask
                x, y, w, h = cv2.boundingRect(np.array(preds[0]))
                l = int(max(w,h))
                x = int(x-(l-w)/2)
                y = int(y-(l-h)*1.5/2)
                face = input_img.copy()[y:y+l,x:x+l]
                print("Crop image!")
                self.SHOW([self.to_plt(input_img),self.to_plt(face)])
        else:
            face = input_img
        
        preds = self.fa.get_landmarks(face)  
        x, y, w, h = cv2.boundingRect(np.array(preds[0][3:14]))
        mask_img = cv2.rectangle(face.copy(), (x, y), (x + w, y + h), (255, 255, 255), -1)
        raw_mask = cv2.rectangle(np.zeros(face.shape,face.dtype), (x, y), (x + w, y + h), (255, 255, 255), -1)

        landmark_cord = preds[0]
        return landmark_cord


    def gen_from_pic(self,path,use_crop=False):
        images,landmark_map,masks, face, landmark_img , raw_mask = self.get_input(path,use_crop)
        outputs = self.gen_result(images.cuda(),landmark_map.cuda(),masks.cuda())
        self.SHOW([self.to_plt(face),raw_mask,landmark_img, np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
        return outputs


    
    def SHOW(self,imagelist,figsize=None,savepath=None):
        num = len(imagelist)
        if figsize==None:
            plt.figure(figsize=(5*num,6))
        else:
            plt.figure(figsize=figsize)
        for i in range(num):
            plt.subplot(1,num,i+1)
            plt.axis('off')
            
            # plt.imshow(cv2.cvtColor(imagelist[i],cv2.COLOR_BGR2RGB))
            plt.imshow(imagelist[i])
            
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
    
    def to_plt(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
torch.cuda.set_device(0)
checkpoint =  "/data/chengbin/code/lafin/checkpoints/celebahq-ae-lmfaceidin-256-512-latent"
api = inpainting_api(checkpoint)
model = api.generator

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

import scipy
from PIL import Image
import torchvision
from skimage.color import rgb2gray

path = "/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/images/"
img_plist = os.listdir(path)
img_path = os.path.join(path,img_plist[1])
print(img_path)
api.gen_from_pic(img_path)

import scipy
import os
from PIL import Image
import torchvision
from skimage.color import rgb2gray
obama_path = '/data/chengbin/dataset/Obama100_1_big'
obama_img_plist = os.listdir(obama_path)
item_list = []
api.load_checkpoint( "/home/public/cb/code/lafin/checkpoints/celebahq-ae-lmfaceidin-256-512-latent_finetune/InpaintingModel_gen.pthe_1")
for i in range(10):
    img_path = os.path.join(obama_path,obama_img_plist[i])
    inputs = api.get_input(img_path) ## out6
    outputs = api.gen_result(inputs[0].cuda(),inputs[1].cuda(),inputs[2].cuda())
    images,landmark_map,masks, face, landmark_img , raw_mask = inputs
#     api.SHOW([api.to_plt(face),api.to_plt(face)/255+raw_mask,landmark_img, np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
    api.SHOW([api.to_plt(face),landmark_img, np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])

    item_list.append([])
    for j in range(len(inputs)):
        item_list[i].append(inputs[j]) 
obama_item_list = item_list


api.load_checkpoint( "/home/public/cb/code/lafin/checkpoints/celebahq-ae-lmfaceidin-256-512-latent_finetune/InpaintingModel_gen.pthe_6")
imageid = 9
lanmarkid = 8
maskid = 9
for lanmarkid in range(10):
    images,landmark_map,masks, face, landmark_img , raw_mask= item_list[imageid][0],item_list[lanmarkid][1],item_list[maskid][2],item_list[imageid][3],item_list[lanmarkid][4], item_list[maskid][5]
    target_face = item_list[lanmarkid][3]
    outputs = api.gen_result(images.cuda(),landmark_map.cuda(),masks.cuda())
    api.SHOW([api.to_plt(face),api.to_plt(face)/255+raw_mask,api.to_plt(target_face)/255+landmark_img, np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
    
    
path = "/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/images/"
img_plist = os.listdir(path)
img_path = os.path.join(path,img_plist[1])
item_list = []
for i in range(10):
    img_path = os.path.join(path,img_plist[i])
    inputs = api.get_input(img_path) ## out6
    outputs = api.gen_result(inputs[0].cuda(),inputs[1].cuda(),inputs[2].cuda())
    images,landmark_map,masks, face, landmark_img , raw_mask = inputs
#     api.SHOW([api.to_plt(face),api.to_plt(face)/255+raw_mask,landmark_img, np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
    api.SHOW([api.to_plt(face),landmark_img, np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
    item_list.append([])
    for j in range(len(inputs)):
        item_list[i].append(inputs[j]) 
        
celeba_item_list = item_list


from scipy.spatial import ConvexHull

obama_path = '/data/chengbin/dataset/Obama100_1'
img_plist = os.listdir(obama_path)
img_path = os.path.join(obama_path,img_plist[1])
lm_list = []
for i in range(10):
    img_path = os.path.join(obama_path,img_plist[i])
    
    lm = api.get_landmark(cv2.imread(img_path)) ## out6
    lm_list.append(lm)
    
path = "/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/images/"
img_plist = os.listdir(path)
img_path = os.path.join(path,img_plist[1])
kp_source = api.get_landmark(cv2.imread(img_path))

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source).volume
        driving_area = ConvexHull(kp_driving_initial).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    #kp_new = {k: v for k, v in kp_driving.items()}
    kp_new = kp_driving

    if use_relative_movement:
        print("use_relative_movement")
        kp_value_diff = (kp_driving - kp_driving_initial)
        kp_value_diff *= adapt_movement_scale
        kp_new = kp_value_diff + kp_source
#         print(kp_new,kp_value_diff,kp_source)
        
#         if use_relative_jacobian:
#             jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
#             kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

landmark_cord = normalize_kp(kp_source,lm_list[0],lm_list[2],adapt_movement_scale=True,use_relative_movement=True)



landmark_cord = normalize_kp(kp_source,lm_list[7],lm_list[9],adapt_movement_scale=True,use_relative_movement=True)

def get_lm_img(landmark_cord,img_size=1024):
    landmark_img = np.zeros((1024,1024,3))
    for i in range(landmark_cord.shape[0]):
        center = (int(landmark_cord[i,0]),int(landmark_cord[i,1]))
        cv2.circle(landmark_img,center, radius=5,color=(255, 255, 255), thickness=-1)
    return landmark_img

api.SHOW([get_lm_img(landmark_cord),get_lm_img(kp_source),get_lm_img(lm_list[0])])


path = "/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/images/"
img_plist = os.listdir(path)
img_path = os.path.join(path,img_plist[1])
api.SHOW([cv2.imread(img_path)])
kp_source = api.get_landmark(cv2.imread(img_path))

obama_path = '/data/chengbin/dataset/Obama100_1'
img_plist = sorted(os.listdir(obama_path))
lm_list = []
for i in range(10):
    img_path = os.path.join(obama_path,img_plist[i])
    input_image = cv2.imread(img_path)
    lm = api.get_landmark(input_image) ## out6
    lm_list.append(lm)

for i in range(len(lm_list)):
    landmark_cord = normalize_kp(kp_source,lm_list[i],lm_list[1],adapt_movement_scale=True,use_relative_movement=True)
    api.SHOW([get_lm_img(kp_source),get_lm_img(landmark_cord),get_lm_img(lm_list[1]),get_lm_img(lm_list[i])])
    
 import torch.nn.functional as F
import torch

torch.cuda.set_device(0)

imageid = 7
lanmarkid = 8
maskid = 7
INPUT_SIZE = 256

kp_initial =  api.get_landmark(obama_item_list[0][3])

for lanmarkid in range(10):
    images,landmark_map,masks, face, landmark_img , raw_mask= celeba_item_list[imageid][0],obama_item_list[lanmarkid][1],celeba_item_list[maskid][2],celeba_item_list[imageid][3],obama_item_list[lanmarkid][4], celeba_item_list[maskid][5]
    target_face = item_list[lanmarkid][3]
#     api.SHOW([np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0))])
    
    
    kp_source = api.get_landmark(face)
    kp_driveing = api.get_landmark(obama_item_list[lanmarkid][3])
    re_landmarks = normalize_kp(kp_source,kp_driveing,kp_initial,adapt_movement_scale=True,use_relative_movement=True)
#     api.SHOW([get_lm_img(kp_source),get_lm_img(re_landmarks),get_lm_img(kp_initial),get_lm_img(kp_driveing)])
    
    
    re_landmarks[:,0] *= (INPUT_SIZE/face.shape[0])
    re_landmarks[:,1] *= (INPUT_SIZE/face.shape[1])
    re_landmarks = (re_landmarks+0.5).astype(np.int16)
    
    landmark_map = torch.zeros(1, INPUT_SIZE,INPUT_SIZE)
    landmark_map[0, re_landmarks[0: INPUT_SIZE, 1], re_landmarks[0: INPUT_SIZE, 0]] = 1
    landmark_map = landmark_map.unsqueeze(dim=0).float()
 

#   landmark_map = F.interpolate(landmark_map, [INPUT_SIZE, INPUT_SIZE], mode='bilinear', align_corners=True)
    
      
#   api.SHOW([np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0))])
    
    outputs = api.gen_result(images.cuda(),landmark_map.cuda(),masks.cuda())
    api.SHOW([api.to_plt(face),api.to_plt(obama_item_list[lanmarkid][3]), np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0)), np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
    
import torch.nn.functional as F
import torch

torch.cuda.set_device(0)

imageid = 4
lanmarkid = 8
maskid = 4
INPUT_SIZE = 256

kp_initial =  api.get_landmark(obama_item_list[0][3])

for lanmarkid in range(10):
    images,landmark_map,masks, face, landmark_img , raw_mask= celeba_item_list[imageid][0],obama_item_list[lanmarkid][1],celeba_item_list[maskid][2],celeba_item_list[imageid][3],obama_item_list[lanmarkid][4], celeba_item_list[maskid][5]
    target_face = item_list[lanmarkid][3]
#     api.SHOW([np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0))])
    
    
#     kp_source = api.get_landmark(face)
#     kp_driveing = api.get_landmark(obama_item_list[lanmarkid][3])
#     re_landmarks = normalize_kp(kp_source,kp_driveing,kp_initial,adapt_movement_scale=True,use_relative_movement=True)
# #     api.SHOW([get_lm_img(kp_source),get_lm_img(re_landmarks),get_lm_img(kp_initial),get_lm_img(kp_driveing)])
    
    
#     re_landmarks[:,0] *= (INPUT_SIZE/face.shape[0])
#     re_landmarks[:,1] *= (INPUT_SIZE/face.shape[1])
#     re_landmarks = (re_landmarks+0.5).astype(np.int16)
    
#     landmark_map = torch.zeros(1, INPUT_SIZE,INPUT_SIZE)
#     landmark_map[0, re_landmarks[0: INPUT_SIZE, 1], re_landmarks[0: INPUT_SIZE, 0]] = 1
#     landmark_map = landmark_map.unsqueeze(dim=0).float()
 

#   landmark_map = F.interpolate(landmark_map, [INPUT_SIZE, INPUT_SIZE], mode='bilinear', align_corners=True)
    
      
#   api.SHOW([np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0))])
    
    outputs = api.gen_result(images.cuda(),landmark_map.cuda(),masks.cuda())
    api.SHOW([api.to_plt(face),api.to_plt(obama_item_list[lanmarkid][3]), np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0)), np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])


