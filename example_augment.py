
from PIL import Image
import math

def stitch_images(outputs, size_in= 256, img_per_row=1,gap=0):
    gap = gap
    nums = len(outputs)
    columns = img_per_row
    rows = int(math.ceil(nums/columns))
#     width, height = outputs[0][:, :, 0].shape 
    width, height = size_in,size_in
#     print(size_in,size_in)
    img = Image.new('RGB', (width * columns + gap * (columns - 1), height * rows + gap * (rows - 1)))

    for ix in range(nums):
        xoffset = int(ix % img_per_row) * width + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height
#         print(xoffset,yoffset)
        im = np.array((outputs[ix])).astype(np.uint8).squeeze()
        im = Image.fromarray(im)
        
        img.paste(im.resize((width, height)), (xoffset, yoffset))
    return img


import cv2
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
img = cv2.imread("examples/images/195579.jpg")
stitch_images([img[...,::-1]])


import torch
from src.augment import AugWrapper

img_t =torch.from_numpy(np.transpose(img,(2,0,1))).unsqueeze(0)/255.0
print(img_t.shape)
augwrapper= AugWrapper(img_t.shape[2])
show_list = []
for i in range(10):
    aug_img_t = augwrapper(img_t,prob=0.7)
    show_list.append(np.transpose(aug_img_t[0].numpy()*255.0,(1,2,0))[...,::-1])
stitch_images(show_list,img_per_row=len(show_list))