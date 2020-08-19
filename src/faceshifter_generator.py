import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .face_modules.model import Backbone


import torch.nn.functional as F
from torch import nn
import torch

import random

from .oneshot_facereencatment import Normal_Encoder

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
#########################################################
def conv(c_in, c_out, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False),
        norm(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )

class conv_transpose(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d):
        super(conv_transpose, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(c_out)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.conv_t(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return x,torch.cat((x, skip), dim=1)


# Multilayer Attributes Encoder
class MAE(nn.Module):
    def __init__(self,c_in=3):
        super(MAE, self).__init__()
        self.conv1 = conv(c_in, 32)
        self.conv2 = conv(32, 64)
        self.conv3 = conv(64, 128)
        self.conv4 = conv(128, 256)
        self.conv5 = conv(256, 512)
        self.conv6 = conv(512, 1024)
        self.conv7 = conv(1024, 1024)

        self.conv_t1 = conv_transpose(1024, 1024)
        self.conv_t2 = conv_transpose(1024, 512)
        self.conv_t3 = conv_transpose(512, 256)
        self.conv_t4 = conv_transpose(256, 128)
        self.conv_t5 = conv_transpose(128, 64)
        self.conv_t6 = conv_transpose(64, 32)

        self.apply(init_weights)

    def forward(self, Xt):    # in 3*256256
        enc1 = self.conv1(Xt) # 32，128，128
        enc2 = self.conv2(enc1) # 64，64，64
        enc3 = self.conv3(enc2) # 128，32，32
        enc4 = self.conv4(enc3) # 256，16，16
        enc5 = self.conv5(enc4) # 512，8，8
        enc6 = self.conv6(enc5) # 1024,4,4
        #print([i.shape for i in [enc1,enc2,enc3,enc4,enc5,enc6]])
        
        z_att1 = self.conv7(enc6)
        dec1, z_att2 = self.conv_t1(z_att1, enc6)
        dec2, z_att3 = self.conv_t2(dec1, enc5)
        dec3, z_att4 = self.conv_t3(dec2, enc4)
        dec4, z_att5 = self.conv_t4(dec3, enc3)
        dec5, z_att6 = self.conv_t5(dec4, enc2)
        dec6, z_att7 = self.conv_t6(dec5, enc1)

        z_att8 = F.interpolate(z_att7, scale_factor=2, mode='bilinear', align_corners=True)

        return z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7, z_att8

########################################
    
class ADD(nn.Module):
    def __init__(self, c_x, c_att, c_id):
        super(ADD, self).__init__()

        self.c_x = c_x

        self.h_conv = nn.Conv2d(in_channels=c_x, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.att_conv1 = nn.Conv2d(in_channels=c_att, out_channels=c_x, kernel_size=1, stride=1, padding=0, bias=True)
        self.att_conv2 = nn.Conv2d(in_channels=c_att, out_channels=c_x, kernel_size=1, stride=1, padding=0, bias=True)

        self.id_fc1 = nn.Linear(c_id, c_x) # c_id 256, c_x 3
        self.id_fc2 = nn.Linear(c_id, c_x)

        self.norm = nn.InstanceNorm2d(c_x, affine=False)

    def forward(self, h, z_att, z_id):
        h_norm = self.norm(h)
        
        att_beta = self.att_conv1(z_att)
        att_gamma = self.att_conv2(z_att)
        
#         print([i.shape for i in [h_norm,att_beta,att_beta]])
#         print(z_id.shape)

        id_beta = self.id_fc1(z_id)
        id_gamma = self.id_fc2(z_id)
#         print(id_beta.shape,id_gamma.shape)

        id_beta = id_beta.reshape(h_norm.shape[0], self.c_x, 1, 1).expand_as(h_norm)
        id_gamma = id_gamma.reshape(h_norm.shape[0], self.c_x, 1, 1).expand_as(h_norm)

        M = torch.sigmoid(self.h_conv(h_norm))
        A = att_gamma * h_norm + att_beta
        I = id_gamma * h_norm + id_beta

        return (torch.ones_like(M).to(M.device) - M) * A + M * I


class ADDResBlk(nn.Module):
    def __init__(self, c_in, c_out, c_att, c_id):
        super(ADDResBlk, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.add1 = ADD(c_in, c_att, c_id)
        self.conv1 = self.conv(c_in, c_in)
        self.add2 = ADD(c_in, c_att, c_id)
        self.conv2 = self.conv(c_in, c_out)

        if c_in != c_out:
            self.add3 = ADD(c_in, c_att, c_id)
            self.conv3 = self.conv(c_in, c_out)

    def forward(self, h, z_att, z_id):
        x = self.add1(h, z_att, z_id)
#         print(x.shape)
        x = self.conv1(x)
        x = self.add1(x, z_att, z_id)
        x = self.conv2(x)
        if self.c_in != self.c_out:
            h = self.add3(h, z_att, z_id)
            h = self.conv3(h)

        return x + h
    
    def conv(self,c_in, c_out):
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False),
        )


class ADDGenerator(nn.Module):
    def __init__(self, c_id=256):
        super(ADDGenerator, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=c_id, out_channels=1024, kernel_size=2, stride=1, padding=0, bias=False)
        self.add1 = ADDResBlk(1024, 1024, 1024, c_id)
        self.add2 = ADDResBlk(1024, 1024, 2048, c_id)
        self.add3 = ADDResBlk(1024, 1024, 1024, c_id)
        self.add4 = ADDResBlk(1024, 512, 512, c_id)
        self.add5 = ADDResBlk(512, 256, 256, c_id)
        self.add6 = ADDResBlk(256, 128, 128, c_id)
        self.add7 = ADDResBlk(128, 64, 64, c_id)
        self.add8 = ADDResBlk(64, 3, 64, c_id)

        self.apply(init_weights)

    def forward(self, z_att, z_id):
        x = self.conv_t(z_id.reshape(z_id.shape[0], -1, 1, 1))
        x = self.add1(x, z_att[0], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add2(x, z_att[1], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add3(x, z_att[2], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add4(x, z_att[3], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add5(x, z_att[4], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add6(x, z_att[5], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add7(x, z_att[6], z_id)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.add8(x, z_att[7], z_id)
        return torch.tanh(x)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class faceshifter_inpaintor(BaseNetwork):
    def __init__(self, image_size=256,IE_pretrained=True,init_weights=True):
        super(faceshifter_inpaintor, self).__init__()
        self.IE_pretrained = IE_pretrained
        if IE_pretrained == True:
            arcface = Backbone(50, 0.6, 'ir_se').cuda()
            arcface.eval()
            arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth'), strict=False)
            self.Idencoder = arcface
        
        self.lm_autoencoder = MAE(c_in=5)
        self.generator = ADDGenerator(c_id=512)
        self.image_size = image_size
        if init_weights:
            self.init_weights()
   
    def Id_forward(self,refimages):
        if self.IE_pretrained == True:
            with torch.no_grad():
                resize_img = F.interpolate(refimages, [112, 112], mode='bilinear', align_corners=True)
                zid, X_feats = self.Idencoder(resize_img)
            return zid, X_feats

    def forward(self,images,refimages):
        batch_size = images.shape[0]
        image_size = self.image_size
        if batch_size<=1:
            print("batch size should be larger than 1")
            exit
        
        z_id, X_feats = self.Id_forward(refimages)
        
        zatt = self.lm_autoencoder(images)
        outputs = self.generator(zatt,z_id)

        out_id, X_feats = self.Id_forward(refimages)
            
        return outputs,z_id,out_id


class faceshifter_fe(BaseNetwork):
    def __init__(self, image_size=256,init_weights=True,same_prob=1):
        super(faceshifter_fe, self).__init__()
        arcface = Backbone(50, 0.6, 'ir_se').cuda()
        arcface.eval()
        arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth'), strict=False)
        self.arcface = arcface
        self.lm_autoencoder = MAE(c_in=1)
        self.generator = ADDGenerator(c_id=512)
        self.image_size = image_size
        if init_weights:
            self.init_weights()
        self.same_prob=same_prob


    def forward(self,images,landmarks,masks):

        batch_size = images.shape[0]
        image_size = self.image_size
        # if batch_size<=1:
        #     print("batch size should be larger than 1")
        #     exit

        is_the_same = (torch.rand(batch_size)< self.same_prob).long().cuda()
        img_index = torch.arange(batch_size).cuda()
        drive_index = img_index*is_the_same.long() + ((img_index+1)%batch_size)*(1-is_the_same).long()
        drive_landmark = landmarks[drive_index]

        with torch.no_grad():
            resize_img = F.interpolate(images, [112, 112], mode='bilinear', align_corners=True)
            zid, X_feats = self.arcface(resize_img)

        zatt = self.lm_autoencoder(drive_landmark)
        outputs = self.generator(zatt,zid)

        with torch.no_grad():
            resize_img = F.interpolate(outputs, [112, 112], mode='bilinear', align_corners=True)
            out_id, X_feats = self.arcface(resize_img)

        return outputs,is_the_same,drive_landmark,zid,out_id

    
    
if __name__ == "__main__":
    # model = faceshifter_sin().cuda()
    # model(torch.rand(2,3,256,256).cuda(),torch.rand(2,1,256,256).cuda(),torch.rand(2,1,256,256).cuda())
    # mae = MAE()
    # zatt = mae(torch.rand(1,3,256,256))
    # addg = ADDGenerator(c_id=512)
    # arcface = Backbone(50, 0.6, 'ir_se')
    # arcface.eval()
    # zid,_ = arcface(torch.rand(1,3,112,112))
    # output = addg(zatt,zid)
    # print(output.shape)

    model = faceshifter_fe().cuda()
    model(torch.rand(1,3,256,256).cuda(),torch.rand(1,1,256,256).cuda(),torch.rand(1,1,256,256).cuda())