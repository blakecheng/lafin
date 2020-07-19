import numpy as np
import matplotlib.pyplot as plt
import pylab
import imageio
import skimage.io
import numpy as np  
import cv2  

def to_plt(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def SHOW(imagelist,figsize=None):
    num = len(imagelist)
    if figsize==None:
        plt.figure(figsize=(6*num,6))
    else:
        plt.figure(figsize=figsize)
    for i in range(num):
        plt.subplot(1,num,i+1)
        plt.axis('off')
        plt.imshow(imagelist[i])
    plt.show()


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
path = 'rawData/1.mp4'
(filepath, tempfilename) = os.path.split(path)
(filename, extension) = os.path.splitext(tempfilename)
img_path = os.path.join(filepath,filename)+"_img"
mkdir(img_path)


cap = cv2.VideoCapture(path)  
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
idx = 0
while(cap.isOpened()):  
    ret, frame = cap.read()
    idx += 1
    if ret == True:
        input_img = frame
        preds = fa.get_landmarks(input_img)  
        if len(preds[0])==68:
            ## mask
            x, y, w, h = cv2.boundingRect(np.array(preds[0]))
            l = int(max(w,h)*1.2)
            x = int(x-(l-w)/2)
            y = int(y-(l-h)/2)
            face = input_img.copy()[y:y+l,x:x+l]
            filename = os.path.join(img_path,"%d.png"% idx)
            cv2.imwrite(filename,face)
            print("writing to %s"% filename)
        else:
            SHOW([to_plt(frame)])