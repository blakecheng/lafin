import torch.nn as nn
import torch.nn.functional as F
import math
from random import random
import torch
import numpy as np

def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random() * delta)
    w_delta = int(random() * delta)
    cropped = tensor[:, :, h_delta:(h_delta + new_width), w_delta:(w_delta + new_width)].clone()
    return F.interpolate(cropped, size=(h, h), mode='bilinear')

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

def get_rot_mat(theta):
    theta = theta*math.pi/180.0
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def random_rotate(x, theta):
    dtype = x.dtype
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid.cuda())
    return x
        

def random_float(lo, hi):
    return lo + (hi - lo) * random()


class AugWrapper(nn.Module):
    def __init__(self, image_size):
        super().__init__()

    def forward(self, images, prob = 0.7, detach = False, is_flip=False):
        if random() < prob:
            random_scale = random_float(0.85, 0.95)
            if is_flip:
                images = random_hflip(images, prob=0.5)
            images = random_crop_and_resize(images, scale = random_scale)
            random_scale_rotate = random_float(-30, 30)
            images = random_rotate(images,theta = random_scale_rotate)
        if detach:
            images.detach_()

        return images
    

    
