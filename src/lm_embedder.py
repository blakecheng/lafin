import os
import glob
import scipy
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
import torchvision
from .oneshot_facereencatment import new_embedder
import torch.optim as optim
import torch.nn as nn
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, landmark_flist, root=None,augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training
        
        self.data = self.load_flist(flist)
        self.landmark_data = self.load_flist(landmark_flist)
        
        if root is not None:
            self.data = [os.path.join(root,i) for i in self.data]
            self.landmark_data = [os.path.join(root,i) for i in self.landmark_data]

        self.input_size = config.INPUT_SIZE

        # in test mode, there's a one-to-one relationship between mask and image
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print("loading the %d th data error"%index)
            while True:
                index = random.randint(0, len(self) - 1)
                try:
                    item = self.load_item(index)
                    break
                except:
                    print("loading the %d th data error"%index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size
        # load image
        img = imread(self.data[index])
        landmark = self.load_lmk([size, size], index, img.shape)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size, centerCrop=True)


        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            landmark[:, 0] = self.input_size - landmark[:, 0]
            landmark = self.shuffle_lr(landmark)


        if self.augment:
            for i in range(3):
                img[i] = (img[i]*np.random.uniform(0.7,1.3))
                img[i][img[i]>1] = 1
                
        return self.to_tensor(img), torch.from_numpy(landmark).long()



    def load_lmk(self, target_shape, index, size_before, center_crop = True):

        imgh,imgw = target_shape[0:2]
        landmarks = np.genfromtxt(self.landmark_data[index])
        landmarks = landmarks.reshape(self.config.LANDMARK_POINTS, 2)

        if self.input_size != 0:
            if center_crop:
                side = np.minimum(size_before[0],size_before[1])
                i = (size_before[0] - side) // 2
                j = (size_before[1] - side) // 2
                landmarks[0:self.config.LANDMARK_POINTS , 0] -= j
                landmarks[0:self.config.LANDMARK_POINTS , 1] -= i

            landmarks[0:self.config.LANDMARK_POINTS ,0] *= (imgw/side)
            landmarks[0:self.config.LANDMARK_POINTS ,1] *= (imgh/side)
        landmarks = (landmarks+0.5).astype(np.int16)

        return landmarks


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = torchvision.transforms.functional.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        #img = scipy.misc.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize([height, width]))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
        print("loading ： %s"%(flist))
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                print("len is : %d "%(len(flist)))
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]
        
        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def shuffle_lr(self, parts, pairs=None):

        if pairs is None:
            if self.config.LANDMARK_POINTS == 68:
                pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                     26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                     34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                     40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                     62, 61, 60, 67, 66, 65]
            elif self.config.LANDMARK_POINTS == 98:
                pairs = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
                         8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39,
                         38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67,
                         66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
        if len(parts.shape) == 3:
            parts = parts[:,pairs,...]
        else:
            parts = parts[pairs,...]
        return parts

def write_log(log_file, logs):
    with open(log_file, 'a') as f:
        f.write('%s\n' % logs)


if __name__ == "__main__":

    from tqdm import tqdm
    class config():
        def __init__(self):
            self.INPUT_SIZE = 256
            self.LANDMARK_POINTS = 68
            self.BATCH_SIZE = 8
            self.LR = 0.00001
            self.BETA1 = 0.0
            self.BETA2 = 0.9
            self.loss = "triplet"
    
    torch.cuda.set_device(5)
    cfg = config()
    train_dataset = Dataset(cfg,
        "/data/chengbin/celeba/celeba-hq/celeba-1024-lafin/images.flist",
        "/data/chengbin/celeba/celeba-hq/celeba-1024-lafin/landmarks.flist")
    
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.BATCH_SIZE,
            num_workers=4,
            shuffle=True,
            drop_last=True
        )

    embedder = pretrained_embedder()
    optimizer = optim.Adam(
                params=embedder.parameters(),
                lr=float(cfg.LR),
                betas=(cfg.BETA1, cfg.BETA2)
            )
    
    
    if cfg.loss == "l1":
        loss_function = nn.L1Loss()
    elif cfg.loss == "triplet":
        loss_function = nn.TripletMarginLoss(margin=0.5)

    for ep in range(1000):
        pbar = tqdm(train_loader,ncols=100)
        for items in pbar:
            optimizer.zero_grad()
            imgs,landmarks = (item.cuda() for item in items)
            landmarks[landmarks >= cfg.INPUT_SIZE] = cfg.INPUT_SIZE - 1
            landmarks[landmarks < 0] = 0
            landmark_map = torch.zeros((cfg.BATCH_SIZE,1,cfg.INPUT_SIZE,cfg.INPUT_SIZE)).cuda()
            for i in range(landmarks.shape[0]):
                landmark_map[i,0,landmarks[i,0:cfg.LANDMARK_POINTS,1],landmarks[i,0:cfg.LANDMARK_POINTS,0]] = 1
            fe,le = embedder(imgs,landmark_map)
            if cfg.loss == "l1":
                loss = loss_function(fe,le)
            elif cfg.loss == "triplet":
                # batch_size = le.shape[0]
                # index = torch.randperm(batch_size).cuda()
                # neg_le = le[index]
                output_anchors = le[0]
                output_positives  = fe[0]
                output_negatives  = fe[1:]
                # print([i.shape for i in [output_anchors,output_positives,output_negatives]])
                output_anchors = output_anchors.unsqueeze(0).expand_as(output_negatives).contiguous()
                output_positives = output_positives.unsqueeze(0).expand_as(output_negatives).contiguous()
                output_negatives = output_negatives.contiguous()

                loss = loss_function(output_anchors,output_positives,output_negatives)
            loss.backward()
            optimizer.step()
            log = "{}\t{}".format(ep,loss.data.cpu())
            write_log("log_{}.dat".format(cfg.loss),log)
            pbar.set_description("loss:{}".format(loss.data.cpu()))
        
        if ep % 20 == 0:
            torch.save({
                'ep': ep,
                'fm_encoder': embedder.Fm_encoder.state_dict(),
                'lm_encoder': embedder.Lm_encoder.state_dict(),
            },"checkpoint_{}.pth".format(cfg.loss)
        )