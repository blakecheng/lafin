import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .face_modules.model import Backbone
from .faceshifter.aei import *


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






class faceshifter_sin(BaseNetwork):
    def __init__(self, image_size=256,init_weights=True):
        super(faceshifter_sin, self).__init__()

        arcface = Backbone(50, 0.6, 'ir_se').cuda()
        arcface.eval()
        arcface.load_state_dict(torch.load('src/saved_models/model_ir_se50.pth'), strict=False)
        self.arcface = arcface
        self.G = AEI_Net(c_id=512)
        self.G.train()
        
        self.image_size = image_size
        if init_weights:
            self.init_weights()


    def forward(self,images,landmarks,masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, landmarks), dim=1)
        
        batch_size = images.shape[0]
        if batch_size<=1:
            print("batch size should be larger than 1")
            exit
        image_size = self.image_size
        
        
        with torch.no_grad():
            resize_img = F.interpolate(images, [112, 112], mode='bilinear', align_corners=True)
            # print(resize_img.shape,images.shape)
            # print( self.arcface)
            
            X_embed, X_feats = self.arcface(resize_img)

        #resize_inputs = F.interpolate(inputs, [112, 112], mode='bilinear', align_corners=True)
        outputs , Xt_attr = self.G(inputs, X_embed)
        
        # index = torch.randperm(batch_size).cuda()
        # ref_x = x[index]
        # ref_masks = masks[index]
        # ref_landmarks = landmarks[index]
        # ref_X_embed, ref_X_feats = X_embed[index], X_feats[index]
        return outputs
    
if __name__ == "__main__":
    model = faceshifter_sin().cuda()
    model(torch.rand(2,3,256,256).cuda(),torch.rand(2,1,256,256).cuda(),torch.rand(2,1,256,256).cuda())