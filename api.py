from src.config import Config
from src.networks import InpaintGenerator
import os
import torch
from src.stylegan2 import stylegan_L2I_Generator,stylegan_L2I_Generator2,stylegan_L2I_Generator3,stylegan_L2I_Generator4,stylegan_L2I_Generator5,stylegan_L2I_Generator_AE
from src.stylegan2 import ref_guided_inpaintor
from src.res_unet import MultiScaleResUNet
import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np  
import cv2  
import face_alignment
import torch.nn.functional as F
from scipy.spatial import ConvexHull
import scipy
from PIL import Image
from skimage.color import rgb2gray

from src.config import Config
from src.networks import InpaintGenerator
import os
import torch
from src.stylegan2 import stylegan_L2I_Generator,stylegan_L2I_Generator2,stylegan_L2I_Generator3,stylegan_L2I_Generator4,stylegan_L2I_Generator5,stylegan_L2I_Generator_AE,stylegan_L2I_Generator_AE_landmark_in
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
import torchvision
import copy
from tqdm import tqdm
import io
import imageio
from src.models import InpaintingModel
from src.warp import get_face_mask
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class inpainting_api():
    def __init__(self,checkpoint):
        config_path = os.path.join(checkpoint,"config.yml")
        weight_path = os.path.join(checkpoint,"InpaintingModel_gen.pth")
        config = Config(config_path)
        torch.cuda.set_device(2)

        self.inpaint_type = "lafin_origin"
        if hasattr(config, 'INPAINTOR'):
            self.inpaint_type = config.INPAINTOR
            
            
        # if  self.inpaint_type == "stylegan2":
        #     print("#####################")
        #     print("USE stylegan generator!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     generator = stylegan_L2I_Generator(image_size=image_size,latent_dim=latent_dim)
        # elif self.inpaint_type == "stylegan2_fixs":
        #     print("#####################")
        #     print("USE stylegan generator, fixstyle!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     generator = stylegan_L2I_Generator2(image_size=image_size,latent_dim=latent_dim)
        # elif self.inpaint_type == "stylegan2_ae": 
        #     print("#####################")
        #     print("USE stylegan generator, AE!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     generator = stylegan_L2I_Generator3(image_size=image_size,latent_dim=latent_dim)
        # elif self.inpaint_type == "stylegan2_ae2":
        #     print("#####################")
        #     print("USE stylegan generator, AE expand!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     num_layers = config.NUM_LAYERS
        #     network_capacity = config.NETWORK_CAPACITY
        #     generator = stylegan_L2I_Generator_AE(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
        # elif self.inpaint_type == "s2_ae_landmark_in":
        #     print("#####################")
        #     print("USE stylegan generator, AE landmark!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     num_layers = config.NUM_LAYERS
        #     network_capacity = config.NETWORK_CAPACITY
        #     generator = stylegan_L2I_Generator_AE_landmark_in(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
        # elif self.inpaint_type == "s2_ae_landmark_and_arcfaceis_in":
        #     print("#####################")
        #     print("USE stylegan generator, AE landmark and arcfaceid!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     num_layers = config.NUM_LAYERS
        #     network_capacity = config.NETWORK_CAPACITY
        #     generator = stylegan_L2I_Generator_AE_landmark_and_arcfaceid_in(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
        # elif self.inpaint_type == "stylegan2_unet": 
        #     print("#####################")
        #     print("USE stylegan generator, unet!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     generator = stylegan_L2I_Generator4(image_size=image_size,latent_dim=latent_dim,fmap_max = 2048)
        # elif self.inpaint_type == "lafin_style": 
        #     print("#####################")
        #     print("USE stylegan generator, lafin_style!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     generator = stylegan_L2I_Generator5(image_size=image_size,latent_dim=latent_dim,fmap_max = 2048)
        # elif self.inpaint_type == "resunet":
        #     print("#####################")
        #     print("USE resunet generator!")
        #     print("#####################\n")
        #     generator = MultiScaleResUNet(in_nc=4,out_nc=3)
        # elif self.inpaint_type == "faceshifter_inpaintor_selfref":
        #     print("#####################")
        #     print("USE faceshifter inpaintor!")
        #     print("#####################\n")
        #     generator = faceshifter_inpaintor()
        # elif self.inpaint_type == "faceshifter_reenactment2":
        #     print("#####################")
        #     print("USE faceshifter inpaintor pairedref!")
        #     print("#####################\n")
        #     generator = faceshifter_reenactment2()
        # elif self.inpaint_type == "faceshifter_reenactment":
        #     print("#####################")
        #     print("USE faceshifter inpaintor!")
        #     print("#####################\n")
        #     generator = faceshifter_reenactment()
        # elif self.inpaint_type == "ref_guided":
        #     print("#####################")
        #     print("USE ref_guided generator!")
        #     print("#####################\n")
        #     image_size = config.INPUT_SIZE
        #     latent_dim = config.LATENT
        #     num_layers = config.NUM_LAYERS
        #     network_capacity = config.NETWORK_CAPACITY
        #     generator = ref_guided_inpaintor(image_size=image_size,latent_dim=latent_dim,network_capacity=network_capacity,num_layers=num_layers)
        # else:
        #     generator = InpaintGenerator()
        self.inpaint_model = InpaintingModel(config)
        generator = self.inpaint_model.generator

        self.INPUT_SIZE = config.INPUT_SIZE
        
        data = torch.load(weight_path,map_location=lambda storage, loc: storage.cuda(2))
        generator.load_state_dict(data["generator"])
        generator.eval()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        self.generator = generator.cuda()
        
    def load_checkpoint(self,weight_path):
        data = torch.load(weight_path)
        self.generator.load_state_dict(data["generator"])
        

    def gen_result(self,images, landmarks, masks, id_images=None,Interpolation=False,alpha=0):
        outputs = self.inpaint_model(images, landmarks, masks,id_images,Interpolation=Interpolation,alpha=alpha)
        # if "stylegan2" in self.inpaint_type:
        #     images_masked = (images * (1 - masks).float()) + masks
        #     inputs = torch.cat((images_masked, landmarks), dim=1)
        #     outputs = self.generator(inputs)
        # elif self.inpaint_type == "s2_ae_landmark_in" :
        #     outputs = self.generator(landmarks)
        # elif self.inpaint_type == "s2_ae_landmark_and_arcfaceis_in":
        #     outputs = self.generator(landmarks,images)
        # elif self.inpaint_type == "faceshifter" :
        #     outputs = self.generator(images,landmarks,masks)
        # elif self.inpaint_type == "faceshifter_inpaintor_selfref":
        #     images_masked = (images * (1 - masks).float()) + masks
        #     inputs = torch.cat((images_masked, landmarks,masks), dim=1)
        #     ref_images = flip(images,dim=1)
        #     outputs,z_id,out_id = self.generator(inputs,ref_images)
        # elif self.inpaint_type == "faceshifter_reenactment2":
        #     batch_size = images.shape[0]
        #     # ref_index = torch.randperm(batch_size).cuda()
        #     ref_index = (torch.arange(batch_size)+1)%batch_size
        #     ref_landmarks = landmarks[ref_index]
        #     ref_images = images[ref_index] 
        #     outputs,z_id,out_id = self.generator(landmarks,ref_images,ref_landmarks)
        #     return ref_landmarks,ref_images,outputs,z_id,out_id
        # elif self.inpaint_type == "faceshifter_reenactment":
        #     batch_size = images.shape[0]
        #     is_the_same = (torch.rand(batch_size)< 0.2).long().cuda()
        #     img_index = torch.arange(batch_size).cuda()
        #     ref_index = img_index*is_the_same.long() + ((img_index+1)%batch_size)*(1-is_the_same).long()
        #     ref_landmarks = landmarks[ref_index]
        #     ref_images = images[ref_index]
        #     outputs,z_id,zatt = self.generator(landmarks,ref_images,ref_landmarks)
        #     return ref_images,ref_landmarks,outputs,z_id,zatt,is_the_same
        # elif "ref_guided" in self.inpaint_type:
        #     outputs = self.generator(images,landmarks,masks)
        # else:
        #     images_masked = (images * (1 - masks).float()) + masks
        #     inputs = torch.cat((images_masked, landmarks), dim=1)
        #     scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
        #                                 mode='bilinear', align_corners=True)
        #     scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
        #                                 mode='bilinear', align_corners=True)
        #     outputs = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter) 
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
    

    def get_input(self,path,use_crop=False,mask_face=False):

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

        if mask_face == True:
            landmarks = np.array(preds[0],dtype = np.int32)
            image_size = (input_img.shape[0], input_img.shape[1])  
            mask = get_face_mask(image_size, landmarks)
            kernel = np.ones((15,15),np.uint8)
            raw_mask = cv2.dilate(mask,kernel,iterations = 1)
            mask_img = cv2.bitwise_and(face, face, mask=raw_mask)
        else:
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
    
    def get_landmark(self,input_img_path,use_crop=False):
        INPUT_SIZE = self.INPUT_SIZE
        input_img = cv2.imread(input_img_path)
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


    
    def SHOW(self,imagelist,figsize=None,savepath=None,is_show=True):
        num = len(imagelist)
        if figsize==None:
            fig = plt.figure(figsize=(5*num,6))
        else:
            fig = plt.figure(figsize=figsize)
        for i in range(num):
            plt.subplot(1,num,i+1)
            plt.axis('off')
            
            # plt.imshow(cv2.cvtColor(imagelist[i],cv2.COLOR_BGR2RGB))
            plt.imshow(imagelist[i])
            
        if savepath is not None:
            plt.savefig(savepath)
        
        if is_show==True:
            plt.show()
        
        return fig
        
    
    def to_plt(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def mouth_inpainting_demo(self,source_img,driving_lmk_dir,max_num=60,INPUT_SIZE=256,fliter_len=1,save_type="gif",fps = 25,filename="debug",only_mouth=False):
        
        def normalize_kp_mouth_anchor(kp_source,kp_driving):
            anchor_source = np.mean(kp_source,axis=0)
            anchor_kp_driving = np.mean(kp_driving,axis=0)
            
            source_delta = kp_source - anchor_source
            kp_driving_delta = kp_driving - anchor_kp_driving

            source_area = ConvexHull(kp_source).volume
            driving_area = ConvexHull(kp_driving).volume
            adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

            target_source = anchor_source + kp_driving_delta*adapt_movement_scale
            
            return target_source

        def normalize_kp_mouth_anchor2(kp_source,kp_driving,kp_driving_initial):
            anchor_source1 = kp_source[0,:]
            anchor_source2 = kp_source[6,:]
            anchor_source = (anchor_source1+anchor_source2)/2
            
            anchor_kp_driving1 = kp_driving[0,:]
            anchor_kp_driving2 = kp_driving[6,:]
            anchor_kp_driving = (anchor_kp_driving1 + anchor_kp_driving2)/2
            
            anchor_kp_driving_initial1 = kp_driving_initial[0,:]
            anchor_kp_driving_initial2 = kp_driving_initial[6,:]
            
            source_delta = kp_source - anchor_source
            kp_driving_delta = kp_driving - anchor_kp_driving
            
            #adapt_movement_scale = np.linalg.norm(anchor_source2 - anchor_source1)/np.linalg.norm(anchor_kp_driving_initial1 - anchor_kp_driving_initial2)
            
            adapt_movement_scale = np.linalg.norm(anchor_source2 - anchor_source1)/np.linalg.norm(anchor_kp_driving_initial1 - anchor_kp_driving_initial2)

            target_source = anchor_source + kp_driving_delta*adapt_movement_scale
            
            return target_source

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

            return kp_new

        class Fliter:
            def __init__(self,num=5,shape=(68,2)):
                self.num = num
                self.count = 0
                self.origin=[]
                self.result = []
                self.sum = np.zeros(shape)
                
            def append(self,data):
                self.origin.append(data)
                self.count +=1
                
                if self.count<=self.num:
                    self.sum += data
                    self.result.append(self.sum/self.count)
                    return self.result[-1]
                
                self.sum = self.sum + data-self.origin[self.count-self.num-1]
                self.result.append(self.sum/self.num)
            
                return self.result[-1]
        
        images,landmark_map,masks, face, landmark_img , raw_mask = self.get_input(source_img)
        
        kp_source = self.get_landmark(source_img)
        if only_mouth:
            my_fliter = Fliter(num=fliter_len,shape=(19,2))
        else:
            my_fliter = Fliter(num=fliter_len,shape=(68,2))
        driving_img_list = sorted(os.listdir(driving_lmk_dir))
        ims = []
        
        if "jpg" in save_type:
            def mkdir(path):
                folder = os.path.exists(path)
                if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(path)        
            mkdir(filename)
            cv2.imwrite(os.path.join(filename,"source.jpg"),cv2.imread(source_img))
            mkdir(os.path.join(filename,"result"))
            mkdir(os.path.join(filename,"retarget"))
            mkdir(os.path.join(filename,"driving"))

        
        for i in tqdm(range(min(len(driving_img_list),max_num))):
            drving_path = os.path.join(driving_lmk_dir,driving_img_list[i])
            face_driving = cv2.imread(drving_path)
            print("loading: %s" % drving_path)
            kp_driving = self.get_landmark(drving_path)
            
            if i ==0:
                kp_driving_initial = kp_driving
                
            
            if only_mouth:
                re_landmarks_mouth = normalize_kp_mouth_anchor2(kp_source[48:67,:],kp_driving[48:67,:],kp_driving_initial[48:67,:])
                re_landmarks_mouth = my_fliter.append(re_landmarks_mouth)
                re_landmarks = copy.copy(kp_source)
                re_landmarks[48:67,:] = re_landmarks_mouth
            else:
                re_landmarks_mouth = normalize_kp(kp_source,kp_driving,kp_driving_initial,True,True,True)
                re_landmarks_mouth = my_fliter.append(re_landmarks_mouth)
                re_landmarks = copy.copy(kp_source)
                re_landmarks = re_landmarks_mouth
                

            re_landmarks[:,0] *= (INPUT_SIZE/face.shape[0])
            re_landmarks[:,1] *= (INPUT_SIZE/face.shape[1])
            re_landmarks = (re_landmarks+0.5).astype(np.int16)
            
            re_landmarks[re_landmarks >= INPUT_SIZE-1] = INPUT_SIZE-1
            re_landmarks[re_landmarks < 0] = 0

            landmark_map = torch.zeros(1, INPUT_SIZE,INPUT_SIZE)
            landmark_map[0, re_landmarks[0: 68, 1], re_landmarks[0: 68, 0]] = 1
            landmark_map = landmark_map.unsqueeze(dim=0).float()
            
            
            outputs = self.gen_result(images.cuda(),landmark_map.cuda(),masks.cuda())
            
            if "jpg" in save_type:
                if "jpg" in save_type:
                    cv2.imwrite(os.path.join(filename,"result","%03d.jpg"%i),255*cv2.cvtColor(np.transpose(outputs[0].data.cpu().numpy(),(1,2,0)),cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(filename,"retarget","%03d.jpg"%i),255* cv2.cvtColor(np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0)), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(filename,"driving","%03d.jpg"%i),face_driving)

          
            
            #outfig = api.SHOW([self.to_plt(face),self.to_plt(face_driving), np.transpose(np.tile(landmark_map.numpy()[0],(3,1,1)),(1,2,0)),np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
            
            outfig = api.SHOW([np.transpose(outputs[0].data.cpu().numpy(),(1,2,0))])
            
            canvas = outfig.canvas
            buffer = io.BytesIO()  # 获取输入输出流对象
            canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
            data = buffer.getvalue()  # 获取流的值
            buffer.write(data)  # 将数据写入buffer
            im = Image.open(buffer)
            ims.append(im)
        
        if  "gif" in save_type:
            print("saving gif")
            kwargs_write = {'fps':fps,'quantizer':'nq'}
            imageio.mimsave(filename+".gif",ims,'GIF-FI',**kwargs_write)
            
            
        if "avi" in save_type:
            print("saving video")
            import warnings
            class VideoWriter:
                def __init__(self, name, width, height, fps=25):
                    # type: (str, int, int, int) -> None
                    if not name.endswith('.avi'):  # 保证文件名的后缀是.avi
                        name += '.avi'
                        warnings.warn('video name should ends with ".avi"')
                    self.__name = name          # 文件名
                    self.__height = height      # 高
                    self.__width = width        # 宽
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 如果是avi视频，编码需要为MJPG
                    self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

                def write(self, frame):
                    if frame.dtype != np.uint8:  # 检查frame的类型
                        raise ValueError('frame.dtype should be np.uint8')
                    # 检查frame的大小
                    row, col, _ = frame.shape
                    if row != self.__height or col != self.__width:
                        warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
                        return
                    self.__writer.write(frame)

                def close(self):
                    self.__writer.release()
            
            
            width = im.size[0]
            height = im.size[1]
            vw = VideoWriter(filename+".avi", width, height,fps)
            count = 0
            for frame in ims:
                image=cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
                vw.write(image)
            vw.close()


if __name__ == "__main__":
    
    ## 整图生成
    #celeba
    torch.cuda.set_device(3)
    checkpoint =  "/home/public/cb/code/lafin/checkpoints/celebahq-ae-lmfaceidin-256-512-latent"
    api = inpainting_api(checkpoint)
    api.load_checkpoint(os.path.join(checkpoint,"InpaintingModel_gen.pthe_1"))
    
    source_img_dir ="/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/images/"
    
    driving_lmk_dir = '/data/chengbin/dataset/host_head_test_mouth_stabel'
    
    # driving_lmk_dir = '/data/chengbin/dataset/deeper_clip/W136_BlendShape_camera_front'
    
    import random 
    index_list = random.sample(range(len(os.listdir(source_img_dir))),500)
    for i in index_list:
        source_img = os.path.join(source_img_dir,os.listdir(source_img_dir)[i])
        api.mouth_inpainting_demo(source_img,driving_lmk_dir,max_num=60,INPUT_SIZE=256,fliter_len=5,save_type=["gif","jpg"], fps=25,filename="result_stylegan2facereenactment_new/img_{}_result".format(i))
    
    
    ## celeba2
    
    # torch.cuda.set_device(3)
    # checkpoint =  "/home/public/cb/code/lafin/checkpoints/celebahq-ae-lmfaceidin-256-512-latent"
    # api = inpainting_api(checkpoint)
    # api.load_checkpoint(os.path.join(checkpoint,"InpaintingModel_gen.pthe_1"))
    # source_img_dir ="/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/images/"
    
    # driving_lmk_dir_all = '/data/chengbin/dataset/test_clip'
    # driving_lmk_dir_list = os.listdir(driving_lmk_dir_all)
    
    # # import random 
    # # index_list = random.sample(range(len(os.listdir(source_img_dir))),200)
    # for i in range(len(driving_lmk_dir_list)):
    #     driving_lmk_dir = os.path.join(driving_lmk_dir_all,driving_lmk_dir_list[i]) 
    #     source_img = os.path.join(source_img_dir,os.listdir(source_img_dir)[20])
    #     api.mouth_inpainting_demo(source_img,driving_lmk_dir,max_num=60,INPUT_SIZE=256,fliter_len=5,save_type=["gif","jpg"], fps=25,filename="result_stylegan2facereenactment_celeba20/test_{}".format(driving_lmk_dir_list[i]))

    
    # torch.cuda.set_device(3)
    # checkpoint =  "/home/public/cb/code/lafin/checkpoints/celebahq-ae-lmfaceidin-256-512-latent"
    # api = inpainting_api(checkpoint)
    # api.load_checkpoint(os.path.join(checkpoint,"InpaintingModel_gen.pthe_1"))

    # source_img_dir ="/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/images/"
    # # driving_lmk_dir = '/data/chengbin/dataset/host_head_test_mouth_stabel'
    # driving_lmk_dir = "/data/chengbin/dataset/test_clip/macron_1"
    # for i in [39]:
    #     source_img = os.path.join(source_img_dir,os.listdir(source_img_dir)[i])
    #     api.mouth_inpainting_demo(source_img,driving_lmk_dir,max_num=60,INPUT_SIZE=256,fliter_len=5,save_type=["gif","avi","jpg"], fps=25,filename="result_stylegan2facereenactment_dug/img_{}".format(i))


    ### 局部补全
    
    
    

