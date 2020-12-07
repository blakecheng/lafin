from PIL import Image
import math
import numpy as np
import cv2
import os
import face_alignment
import torch
from skimage import transform as trans

torch.cuda.set_device(3)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

data_root = "/opt/mnt/cb/dataset/FFHQ/ffhq-lafin/images"
img_list = os.listdir(data_root)

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    
    tform.estimate(lmk, src)
    M = tform.params[0:2, :]

    return M


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def landmark_68_to_5(landmarks68):
    landmarks5 = []
    landmarks5.append(landmarks68[36][:2])
    landmarks5.append(landmarks68[45][:2])
    landmarks5.append(landmarks68[33][:2])
    landmarks5.append(landmarks68[48][:2])
    landmarks5.append(landmarks68[54][:2])
    return np.array(landmarks5)


# warped=norm_crop(cv2.resize(img,(112,112)),landmarks5*112.0/1024.0)

img = cv2.imread(os.path.join(data_root,img_list[34185]))
landmarks68 = fa.get_landmarks(img)[0]
landmarks5 = landmark_68_to_5(landmarks68)
warped=norm_crop(img,landmarks5)


