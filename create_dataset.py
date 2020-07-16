import os
import subprocess
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm 
import yaml
import argparse

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

def create_dataset(nums,img_dir,mask_dir,landmark_dir,output_path):
    files = os.listdir(img_dir)
    files = [os.path.splitext(i)[0] for i in files]
    file_num = len(files)
    file_nums = [int(i*file_num) for i in nums]
    print("file nums:", file_nums)
    
    trainfiles = files[:file_nums[0]]
    valfiles = files[file_nums[0]+1: file_nums[0]+file_nums[1]]
    testfiles = files[file_nums[0]+file_nums[1]+1:]
    
    filedict = {
        "train": trainfiles,
        "val" : valfiles,
        "test": testfiles
    }
    
    
    mkdir(output_path)
    mkdir(os.path.join(output_path,"train"))
    mkdir(os.path.join(output_path,"val"))
    mkdir(os.path.join(output_path,"test"))
    
    
    
    def create_flist(img_dir, mask_dir, landmark_dir, input_files,output_path):
        images = []
        masks = []
        landmarks =[]
        for file in input_files:
            images.append(os.path.join(img_dir, file)+".jpg")
            masks.append(os.path.join(mask_dir, file)+".png")
            landmarks.append(os.path.join(landmark_dir, file)+".txt")
        images = sorted(images)
        masks = sorted(masks)
        landmarks = sorted(landmarks)

    #     debuglist = [images[0],masks[0],landmarks[0]]
    #     print([os.path.exists(file) for file in debuglist])

        np.savetxt(output_path+"/images.flist", images, fmt='%s')
        np.savetxt(output_path+"/masks.flist", masks, fmt='%s')
        np.savetxt(output_path+"/landmarks.flist", landmarks, fmt='%s')
        
    create_flist(img_dir,mask_dir,landmark_dir,filedict['train'],os.path.join(output_path,"train"))
    create_flist(img_dir,mask_dir,landmark_dir,filedict['val'],os.path.join(output_path,"val"))
    create_flist(img_dir,mask_dir,landmark_dir,filedict['test'],os.path.join(output_path,"test"))
    
def create_config(dataset_path,target_path,example_path='config.yml'):
   
    mkdir(target_path)

    f = open(example_path) 
    config = yaml.load(f) 

    config['TRAIN_INPAINT_IMAGE_FLIST'] = dataset_path+"/train/images.flist" 
    config['TRAIN_INPAINT_LANDMARK_FLIST'] = dataset_path+"/train/landmarks.flist" 
    config['TRAIN_MASK_FLIST'] = dataset_path+"/train/masks.flist"

    config['TEST_INPAINT_IMAGE_FLIST'] = dataset_path+"/test/images.flist" 
    config['TEST_INPAINT_LANDMARK_FLIST'] = dataset_path+"/test/landmarks.flist" 
    config['TEST_MASK_FLIST'] = dataset_path+"/test/masks.flist"

    config['VAL_INPAINT_IMAGE_FLIST'] = dataset_path+"/val/images.flist" 
    config['VAL_INPAINT_LANDMARK_FLIST'] = dataset_path+"/val/landmarks.flist" 
    config['VAL_MASK_FLIST'] = dataset_path+"/val/masks.flist"

    fr = open(os.path.join(target_path,'config.yml'), 'w')
    yaml.dump(config, fr)
    fr.close()
    print(target_path)
    print("done")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pic', default='/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin-debug/' ,type=str, help='path to the dataset')
    parser.add_argument('--dataset', default="datasets/celeba1024-debug" ,type=str, help='path to the file list')
    parser.add_argument('--checkpoint', default="checkpoints/celeba1024-debug" ,type=str, help='path to the file list')
    parser.add_argument('--config',default='config.yml',type = str, help = 'path to the config')
    args = parser.parse_args()
    
    nums=[0.9,0.05,0.05]
    pic_path = args.pic
    dataset_path = args.dataset
    checkpoint_path = args.checkpoint

    img_dir= os.path.join(pic_path,"images")
    mask_dir= os.path.join(pic_path,"masks")
    landmark_dir = os.path.join(pic_path,"landmarks")

    create_dataset(nums,img_dir,mask_dir,landmark_dir,dataset_path)
    create_config(dataset_path,checkpoint_path,example_path=args.config)
    print("dataset locate：%s ,checkpoint locate : %s "%(dataset_path,checkpoint_path))
    
    

  
