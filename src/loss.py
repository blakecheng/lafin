import torch
import torch.nn as nn
import torchvision.models as models
from .networks import MobileNetV2
from .face_modules.model import Backbone
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import os
# from torch.utils.data.dataloader import pin_memory_batch

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()
            
        print("loss type: %s"% type)

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class New_AdversarialLoss(nn.Module):
    def __init__(self, gan_type):
        super().__init__()
        self.gan_type = gan_type

    def get_dis_preds(self, real_score, fake_score):
        if self.gan_type == 'gan':
            real_pred = real_score
            fake_pred = fake_score
        elif self.gan_type == 'rgan':
            real_pred = real_score - fake_score
            fake_pred = fake_score - real_score
        elif self.gan_type == 'ragan':
            real_pred = real_score - fake_score.mean()
            fake_pred = fake_score - real_score.mean()
        else:
            raise Exception('Incorrect `gan_type` argument')
        return real_pred, fake_pred

    def forward(self, fake_score_G,fake_score_D,real_score):

        real_pred, fake_pred_D = self.get_dis_preds(real_score, fake_score_D)
        _, fake_pred_G = self.get_dis_preds(real_score, fake_score_G)

        loss_D = torch.relu(1. - real_pred).mean() + torch.relu(1. + fake_pred_D).mean()  # TODO: check correctness

        if self.gan_type == 'gan':
            loss_G = -fake_pred_G.mean()
        elif 'r' in self.gan_type:
            loss_G = torch.relu(1. + real_pred).mean() + torch.relu(1. - fake_pred_G).mean()
        else:
            raise Exception('Incorrect `gan_type` argument')

        return loss_G, loss_D



class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x = F.interpolate(x, [256, 256], mode='bilinear', align_corners=True)
        y = F.interpolate(y, [256, 256], mode='bilinear', align_corners=True)
  
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



# class PerceptualLoss(nn.Module):
#     r"""
#     Perceptual loss, VGG-based
#     https://arxiv.org/abs/1603.08155
#     https://github.com/dxyang/StyleTransfer/blob/master/utils.py
#     """

#     def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
#         super(PerceptualLoss, self).__init__()
#         self.add_module('vgg', VGG19())
#         self.criterion = torch.nn.L1Loss()
#         self.weights = weights

#     def __call__(self, x, y):
#         # Compute features
#         x_vgg, y_vgg = self.vgg(x), self.vgg(y)

#         content_loss = 0.0
#         content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
#         content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
#         content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
#         content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
#         content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

#         return content_loss

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss,self).__init__()

        arcface = Backbone(50, 0.6, 'ir_se').cuda()
        arcface.eval()
        arcface.load_state_dict(torch.load('saved_models/model_ir_se50.pth'), strict=False)
        self.arcface = arcface
        self.arcface.eval()
        
    def forward(self,img_true,img_pred):
        with torch.no_grad():
            self.arcface.eval()
            img_true = F.interpolate(img_true, [112, 112], mode='bilinear', align_corners=True)
            img_pred = F.interpolate(img_pred, [112, 112], mode='bilinear', align_corners=True)
            feature_ture, _ = self.arcface(img_true)
            feature_pred, _ = self.arcface(img_pred)
        return  (1 - torch.cosine_similarity(feature_ture, feature_pred, dim=1)).mean()




class PerceptualLoss(nn.Module):
    def __init__(self, vgg_weights_dir="saved_models", net='face', normalize_grad=False):
        super().__init__()
        self.normalize_grad = normalize_grad

        if net == 'pytorch':
            model = torchvision.models.vgg19(pretrained=True).features

            mean = torch.tensor([0.485, 0.456, 0.406])
            std  = torch.tensor([0.229, 0.224, 0.225])

            num_layers = 30

        elif net == 'caffe':
            vgg_weights = torch.load(os.path.join(vgg_weights_dir, 'vgg19-d01eb7cb.pth'))

            map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
            vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])

            model = torchvision.models.vgg19()
            model.classifier = nn.Sequential(Flatten(), *model.classifier._modules.values())

            model.load_state_dict(vgg_weights)

            model = model.features

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

            num_layers = 30

        elif net == 'face':
            # Load caffe weights for VGGFace, converted from
            # https://media.githubusercontent.com/media/yzhang559/vgg-face/master/VGG_FACE.caffemodel.pth
            # The base model is VGG16, not VGG19.
            model = torchvision.models.vgg16().features
            model.load_state_dict(torch.load(os.path.join(vgg_weights_dir, 'vgg_face_weights.pth')))

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

            num_layers = 30

        else:
            raise ValueError(f"Unknown type of PerceptualLoss: expected '{{pytorch,caffe,face}}', got '{net}'")

        self.register_buffer('mean', mean[None, :, None, None])
        self.register_buffer('std' ,  std[None, :, None, None])

        self.net = net


        layers_avg_pooling = []

        for weights in model.parameters():
            weights.requires_grad = False

        for module in model.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                layers_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers_avg_pooling.append(module)

            if len(layers_avg_pooling) >= num_layers:
                break

        layers_avg_pooling = nn.Sequential(*layers_avg_pooling)

        self.model = layers_avg_pooling

    def normalize_inputs(self, x):
        return (x - self.mean) / self.std

    def forward(self, input, target,landmarks=None):

        if self.net == "face":
            if landmarks is not None:
                bboxes_estimate = compute_bboxes_from_keypoints(landmarks)
                # convert bboxes from [0; 1] to pixel coordinates
                h, w = input.shape[2:]
                bboxes_estimate[:, 0:2] *= h
                bboxes_estimate[:, 2:4] *= w
            else:
                crop_factor = 1 / 1.8
                h, w = input.shape[2:]

                t = h * (1 - crop_factor) / 2
                l = w * (1 - crop_factor) / 2
                b = h - t
                r = w - l

                bboxes_estimate = torch.empty((1, 4), dtype=torch.float32, device=input.device)
                bboxes_estimate[0].copy_(torch.tensor([t, b, l, r]))
                bboxes_estimate = bboxes_estimate.expand(len(input), 4)

            fake_rgb_cropped = crop_and_resize(input, bboxes_estimate)
            real_rgb_cropped = crop_and_resize(target, bboxes_estimate)

        input = (input + 1) / 2
        target = (target.detach() + 1) / 2
        loss = 0
        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)

        for layer in self.model:
            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                if self.normalize_grad:
                    pass
                else:
                    loss = loss + F.l1_loss(features_input, features_target)

        return loss

def crop_and_resize(images, bboxes, target_size=None):
    """
    images: B x C x H x W
    bboxes: B x 4; [t, b, l, r], in pixel coordinates
    target_size (optional): tuple (h, w)

    return value: B x C x h x w

    Crop i-th image using i-th bounding box, then resize all crops to the
    desired shape (default is the original images' size, H x W).
    """
    t, b, l, r = bboxes.t().float()
    batch_size, num_channels, h, w = images.shape

    affine_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=images.device)
    affine_matrix[:, 0, 0] = (r-l) / w
    affine_matrix[:, 1, 1] = (b-t) / h
    affine_matrix[:, 0, 2] = (l+r) / w - 1
    affine_matrix[:, 1, 2] = (t+b) / h - 1

    output_shape = (batch_size, num_channels) + (target_size or (h, w))
    try:
        grid = torch.affine_grid_generator(affine_matrix, output_shape, False)
    except TypeError: # PyTorch < 1.4.0
        grid = torch.affine_grid_generator(affine_matrix, output_shape)
    return torch.nn.functional.grid_sample(images, grid, 'bilinear', 'reflection')

def compute_bboxes_from_keypoints(keypoints):
    """
    keypoints: B x 68*2

    return value: B x 4 (t, b, l, r)

    Compute a very rough bounding box approximate from 68 keypoints.
    """
    x, y = keypoints.float().view(-1, 68, 2).transpose(0, 2)

    face_height = y[8] - y[27]
    b = y[8] + face_height * 0.2
    t = y[27] - face_height * 0.47

    midpoint_x = (x.min() + x.max()) / 2
    half_height = (b - t) * 0.5
    l = midpoint_x - half_height
    r = midpoint_x + half_height

    return torch.stack([t, b, l, r], dim=1)



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x):
        height = x.size()[2]
        width = x.size()[3]
        tv_h = torch.div(torch.sum(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])),(x.size()[0]*x.size()[1]*(height-1)*width))
        tv_w = torch.div(torch.sum(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])),(x.size()[0]*x.size()[1]*(height)*(width-1)))
        return tv_w+tv_h


class Landmark_loss(nn.Module):
    def __init__(self,points_num=68):
        super(Landmark_loss,self).__init__()
        self.points_num = points_num
        self.lm_detector = MobileNetV2(points_num=points_num).cuda()
        lm_weight = torch.load("saved_models/landmark_detector.pth")
        self.lm_detector.load_state_dict(lm_weight['detector'])
        self.lm_detector.eval()
        
    def forward(self,img_true,img_pred):
        img_true = F.interpolate(img_true, [256, 256], mode='bilinear', align_corners=True)
        img_pred = F.interpolate(img_pred, [256, 256], mode='bilinear', align_corners=True)
        landmark_true, landmark_pred = self.lm_detector(img_true),self.lm_detector(img_pred)
        landmark_loss = torch.norm((landmark_true-landmark_pred).reshape(-1,self.points_num*2),2,dim=1,keepdim=True)
        return torch.mean(landmark_loss)






class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss,self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


