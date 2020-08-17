from src.faceshifter_generator import faceshifter_fe
from src.faceshifter_discriminator import MultiscaleDiscriminator
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch
import time
import cv2
from tqdm import tqdm
from src.lm_embedder import *
torch.cuda.set_device(6)

batch_size = 1
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 10
save_epoch = 1
model_save_path = './saved_models/'

device = torch.device('cuda')

Gmodel = faceshifter_fe().cuda()
Dmodel = MultiscaleDiscriminator(3).cuda()

Gmodel.train()
Dmodel.train()

opt_G = optim.Adam(Gmodel.parameters(), lr=lr_G, betas=(0, 0.999))
opt_D = optim.Adam(Dmodel.parameters(), lr=lr_D, betas=(0, 0.999))

MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()

def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()

class config():
    def __init__(self):
        self.INPUT_SIZE = 256
        self.LANDMARK_POINTS = 68
        self.BATCH_SIZE = 4
        self.LR = 0.00001
        self.BETA1 = 0.0
        self.BETA2 = 0.9

cfg = config()
# train_dataset = Dataset(cfg,
#     "/data/chengbin/celeba/celeba-hq/celeba-1024-lafin/images.flist",
#     "/data/chengbin/celeba/celeba-hq/celeba-1024-lafin/landmarks.flist")

train_dataset = Dataset(cfg,
    "/home/chengbin/code/lafin_school/checkpoints/Obama_face/images.flist",
    "/home/chengbin/code/lafin_school/checkpoints/Obama_face/landmarks.flist")

train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.BATCH_SIZE,
            num_workers=4,
            shuffle=True,
            drop_last=True)


def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()


for epoch in range(0,max_epoch):
    pbar = tqdm(train_loader,ncols=100)
    for items in pbar:
        start_time = time.time()
        imgs,landmarks = (item.cuda() for item in items)
        landmarks[landmarks >= cfg.INPUT_SIZE] = cfg.INPUT_SIZE - 1
        landmarks[landmarks < 0] = 0
        landmark_map = torch.zeros((cfg.BATCH_SIZE,1,cfg.INPUT_SIZE,cfg.INPUT_SIZE)).cuda()

        batch_size = imgs.shape[0]

        for i in range(landmarks.shape[0]):
            landmark_map[i,0,landmarks[i,0:cfg.LANDMARK_POINTS,1],landmarks[i,0:cfg.LANDMARK_POINTS,0]] = 1


        # train G
        opt_G.zero_grad()
        outputs,is_the_same,drive_landmark,zid,out_id = Gmodel(imgs,landmark_map,landmark_map)
        
        L_id = (1 - torch.cosine_similarity(zid, out_id, dim=1)).mean()

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(imgs - outputs, 2).reshape(batch_size, -1), dim=1) * is_the_same.float()) / (is_the_same.sum().float() + 1e-6)
        L_rec =0
        Di = Dmodel(outputs)
        L_adv = 0

        for di in Di:
            L_adv += hinge_loss(di[0], True)
        
        l_adv = 1
        l_id = 1
        l_rec = 10

        lossG = l_adv*L_adv + l_id*L_id + l_rec*L_rec

        lossG.backward()
        opt_G.step()

        # train D
        opt_D.zero_grad()
        fake_D = Dmodel(outputs.detach())
        loss_fake = 0
        for di in fake_D:
            loss_fake += hinge_loss(di[0], False)

        true_D = Dmodel(imgs)
        loss_true = 0
        for di in true_D:
            loss_true += hinge_loss(di[0], True)

        lossD = 0.5*(loss_true.mean() + loss_fake.mean())
        lossD.backward()
        opt_D.step()
        batch_time = time.time() - start_time

        pbar.set_description("gloss:{}  dloss:{}".format(lossG.data.cpu(),lossD.data.cpu()))
    

    if epoch % 20 == 0:
            torch.save({
                'ep': epoch,
                'fm_encoder': Gmodel.state_dict(),
                'lm_encoder': Dmodel.state_dict(),
            },"checkpoint.pth"
        )