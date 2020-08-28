import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np  
import cv2  
import face_alignment
from multiprocessing import Pool
## example :
# python create_face_dataset.py 
# --path /data/qg_data/video/data1-data2-clipped/B73A8713_78.mov 
# --output /data/qg_data/video/data1-data2-clipped-img/B73A8713_78_face/images


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,default = '/data/chengbin/dataset/host_img', help='path to the video')
parser.add_argument('--output', type=str, default = '/data/chengbin/dataset/host_face_lafin', help='path to the face image')
parser.add_argument('--is_video',type=bool)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
path = args.path
# (filepath, tempfilename) = os.path.split(path)
# (filename, extension) = os.path.splitext(tempfilename)
img_path = args.output
mkdir(img_path)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def get_landmark(input_img):
    preds = fa.get_landmarks(input_img)
    if len(preds[0])==68:
        ## mask
        x, y, w, h = cv2.boundingRect(np.array(preds[0]))
        l = int(max(w,h)*1.7)
        x = max(0,int(x-(l-w)/2))
        y = max(0, int(y-(l-h)*1.5/2))
        face = input_img.copy()[y:y+l,x:x+l]
        filename = os.path.join(img_path,img)
        cv2.imwrite(filename,face)
        print("writing to %s"% filename)
    else:
        print("Error occured ：{}".format(idx))

def main():
    if args.is_video == True:
        cap = cv2.VideoCapture(path)  
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        idx = 0
        while(cap.isOpened()):  
            ret, frame = cap.read()
            idx += 1
            # if idx%15 is not 0:
            #     continue
            
            if ret == True:
                input_img = frame
                preds = fa.get_landmarks(input_img)  
                if len(preds[0])==68:
                    ## mask
                    x, y, w, h = cv2.boundingRect(np.array(preds[0]))
                    l = int(max(w,h)*1.7)
                    x = max(0,int(x-(l-w)/2))
                    y = max(0, int(y-(l-h)*1.5/2))
                    face = input_img.copy()[y:y+l,x:x+l]
                    filename = os.path.join(img_path,"%d.jpg"% idx)
                    cv2.imwrite(filename,face)
                    print("writing to %s"% filename)
                else:
                    print("Error occured ：{}".format(idx))
            else:
                break
        print("end")
    else:
        imgs = os.listdir(path)
        for img in imgs:
            try:
                input_img = cv2.imread(os.path.join(path,img))
                get_landmark(input_img)
            except:
                print("Error when detecting %s"%img)

if __name__ == '__main__':
    main()
       
        
                          


    # %%
