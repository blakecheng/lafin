import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.functions import draw_landmarks
from utils.render import render
from utils.depth import depth

import matplotlib.pyplot as plt
import os

root = "/opt/mnt/cb/dataset/FFHQ/ffhq-lafin/images"
save_dir = "/opt/mnt/cb/dataset/FFHQ/ffhq-lafin/render"
img_list= os.listdir(root)

from tqdm import tqdm
from utils.pncc import pncc

cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX
    
    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    tddfa = TDDFA(gpu_mode=False, **cfg)
    face_boxes = FaceBoxes()

for i in tqdm(range(len(img_list))):
    img_fp = os.path.join(root,img_list[i])
    img = cv2.imread(img_fp)
    boxes = face_boxes(img)
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    out=render(img, ver_lst, tddfa.tri, show_flag=False, with_bg_flag=False,wfp=os.path.join(save_dir,img_list[i]))
