import os
import sys
sys.path.insert(0, "./3DDFA_V2")
# os.chdir("3DDFA_V2")
import glob

from tqdm import tqdm
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.functions import draw_landmarks
from utils.render import render
from utils.depth import depth

import matplotlib.pyplot as plt
from utils.pncc import pncc

# load config
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


datapath = glob.glob("/data/chengbin/celeba/celeba-hq/celeba-256/*.jpg")
result_path = "/data/chengbin/celeba/celeba-hq/celeba-256-3d/pncc"
for data in tqdm(datapath):
    root,filename = os.path.split(data)
    basename,ext = os.path.splitext(filename)
    img = cv2.imread(data)
    boxes = face_boxes(img)
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    pncc(img, ver_lst, tddfa.tri, show_flag=False, with_bg_flag=False,wfp=os.path.join(result_path,filename))