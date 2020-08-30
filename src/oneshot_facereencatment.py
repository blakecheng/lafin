import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils
import math

class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)



class fr_Encoder(nn.Module):
    def __init__(self, in_c, depth = 5, num_features=64, activation=F.leaky_relu, nf_max = 512):
        super(fr_Encoder, self).__init__()
        self.num_features = num_features
        self.activation = activation
        
        ch = [min(nf_max, num_features * (2 ** i)) for i in range(depth)]
        
        blocklist = []
        init_block = OptimizedBlock(in_c, num_features) # 128
        blocklist.append(init_block)
        
        for i in range(depth-1):
            block = Block(ch[i], ch[i+1],
                            activation=activation, downsample=True) 
            blocklist.append(block)
        
        self.blocks = nn.ModuleList(blocklist)
        self.L = utils.spectral_norm(nn.Linear(ch[-1], 512))
        
        self.init_weights()
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        h = self.activation(x)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        h = self.activation(h)   # 512
        out = self.L(h)
        
        return out
    
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


class Normal_Encoder(nn.Module):
    def __init__(self, in_c, depth = 5, num_features=64, activation=F.leaky_relu, nf_max = 512):
        super(Normal_Encoder, self).__init__()
        self.num_features = num_features
        self.activation = activation
        
        ch = [min(nf_max, num_features * (2 ** i)) for i in range(depth)]
        
        blocklist = []
        init_block = OptimizedBlock(in_c, num_features) # 128
        blocklist.append(init_block)
        
        for i in range(depth-1):
            block = Block(ch[i], ch[i+1],
                            activation=activation, downsample=True) 
            blocklist.append(block)
        
        self.blocks = nn.ModuleList(blocklist)
        self.L = utils.spectral_norm(nn.Linear(ch[-1], 512))
        
        self.init_weights()
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        h = self.activation(x)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        h = self.activation(h)   # 512
        out = self.L(h)
        
        return out
    
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
        
        



class new_embedder(nn.Module):
    def __init__(self):
        super(new_embedder, self).__init__()
        self.Lm_encoder = Normal_Encoder(1,3).cuda()
        self.Fm_encoder = Normal_Encoder(3,5).cuda()
    
    def forward(self,imgs,landmarks):
        return self.Fm_encoder(imgs),self.Lm_encoder(landmarks)



    
