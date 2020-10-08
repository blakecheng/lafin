import os
import subprocess
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm 
import yaml
import argparse
import face_alignment 

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

def command(cmd="ls"):
    d=subprocess.getstatusoutput(str(cmd))
    if d[0] == 0:
        print("Command success: {}".format(cmd))
        print("Output: \n {}".format(d[1]))
    else:
        print("Command fail: {}".format(cmd))
        print("Error message: \n {}".format(d[1]))  

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
    
    
def to_plt(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 


def get_face_landmarks(image, face_detector, shape_predictor):
    """
    获取人脸标志，68个特征点
    :param image: image
    :param face_detector: dlib.get_frontal_face_detector
    :param shape_predictor: dlib.shape_predictor
    :return: np.array([[],[]]), 68个特征点
    """
    dets = face_detector(image, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        return None
    shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return face_landmarks  

def get_mouth_mask(image_size, face_landmarks):
    """
    获取人脸掩模
    :param image_size: 图片大小
    :param face_landmarks: 68个特征点
    :return: image_mask, 掩模图片
    """
    mask = np.zeros(image_size, dtype=np.uint8)
#     points = np.concatenate([face_landmarks[48:54], face_landmarks[26:17:-1]])
    points = np.concatenate([face_landmarks[48:60]])
    cv2.fillPoly(img=mask, pts=[points], color=255)

    return mask 

def create_mask(path="datasets/debug/", img_dir = 'datasets/debug/images/',
                mask_dir='datasets/debug/masks/', 
                img_flist = '/datasets/debug/images.flist',
                mask_flist = "/datasets/debug/masks.flist",
                img_num = None):
    from src.warp import get_face_mask

    mkdir(img_dir)
    mkdir(mask_dir)
    img_list = os.listdir(path)
    
    if img_num is not None:
        img_list = np.random.choice(img_list,img_num)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    for file_name in tqdm(img_list):
       
        file_name_no_ext = os.path.splitext(file_name)[0]
        if os.path.exists(os.path.join(mask_dir,file_name_no_ext+'.png')):
            continue

        
        input_img = cv2.imread(os.path.join(path,file_name))
        
        try:
            if input_img==None:
                print("Error occured when loading: \n %s" % os.path.join(path,file_name))
                continue
        except:
            pass

        try:
            preds = fa.get_landmarks(input_img)
        except:
            continue

        if preds==None:
            print("Can't detect landmark from pic %s"%(file_name))
            continue

        landmarks = np.array(preds[0],dtype = np.int32)
        image_size = (input_img.shape[0], input_img.shape[1])  
        mask = get_face_mask(image_size, landmarks)
        kernel = np.ones((15,15),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        
        # x, y, w, h = cv2.boundingRect(np.concatenate((np.array(preds[0][3:14,:]),np.array(preds[0][48:67,:])),axis=0))
        # mask_img = cv2.rectangle(input_img.copy(), (x, y), (x + w, y + h), (255, 255, 255), -1)
        # mask = cv2.rectangle(np.zeros(input_img.shape,input_img.dtype), (x, y), (x + w, y + h), (255, 255, 255), -1)
            
        cv2.imwrite(os.path.join(mask_dir, file_name_no_ext+'.png'),mask)
        
    path_to_mask = mask_dir
    path_to_mask_flist= mask_flist
    create_mask_flist_cmd = "python ./scripts/flist.py --path {} --output {}".format(path_to_mask,path_to_mask_flist)
    command(create_mask_flist_cmd)
    
    print("images : {} , masks : {} , landmarks : {} ".format(len(os.listdir(img_dir)),len(os.listdir(mask_dir)),len(os.listdir(landmark_dir))))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pic', default='/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin-debug/' ,type=str, help='path to the dataset')
    parser.add_argument("--raw",default='/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/' ,type=str, help='path to the raw')
    args = parser.parse_args()
    
    
    img_dir =  os.path.join(args.pic,"images")
    img_flist = os.path.join(args.pic,'images.flist')
    mask_dir= os.path.join(args.pic,"masks_face")
    mask_flist = os.path.join(args.pic,"masks.flist")
    
    create_mask(args.raw,img_dir,mask_dir,img_flist,mask_flist)
    
    

  
