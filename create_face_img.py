import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np  
import cv2  
import face_alignment

## example :
# python create_face_dataset.py 
# --path /data/qg_data/video/data1-data2-clipped/B73A8713_78.mov 
# --output /data/qg_data/video/data1-data2-clipped-img/B73A8713_78_face/images


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the video')
parser.add_argument('--output', type=str, help='path to the face image')
args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = "2"
path = args.path
# (filepath, tempfilename) = os.path.split(path)
# (filename, extension) = os.path.splitext(tempfilename)
img_path = args.output
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
            filename = os.path.join(img_path,"%d.jpg"% idx)
            cv2.imwrite(filename,face)
            print("writing to %s"% filename)
        else:
            print("Error occured ：{}".format(idx))
    else:
        break
print("end")