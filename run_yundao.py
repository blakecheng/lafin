import os
from datetime import datetime
import time
import sys
import threading
import yaml


def copy_file(obs_path, cache_path):
    if not os.path.exists(os.path.dirname(cache_path)): os.makedirs(os.path.dirname(cache_path))
    print('start copy {} to {}: {}'.format(obs_path, cache_path, datetime.now().strftime("%m-%d-%H-%M-%S")))
    mox.file.copy(obs_path, cache_path)
    print('end copy {} to cache: {}'.format(obs_path, datetime.now().strftime("%m-%d-%H-%M-%S")))


def copy_dataset(obs_path, cache_path):
    if not os.path.exists(cache_path): os.makedirs(cache_path)
    print('start copy {} to {}: {}'.format(obs_path, cache_path, datetime.now().strftime("%m-%d-%H-%M-%S")))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    mox.file.copy_parallel(obs_path, cache_path)
    print('end copy {} to cache: {}'.format(obs_path, datetime.now().strftime("%m-%d-%H-%M-%S")))


def get_checkpoint(checkpoint_path, s3chekpoint_path):
    def get_time(i):
        return min(600, i)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    i = 1
    while True:
        i = i+1
        time.sleep(get_time(i))
        print("runtime : {} min ".format((time.time() - start) / 60))
        copy_dataset(checkpoint_path, s3chekpoint_path)

def show_nvidia():
    os.system("nvidia-smi")
    while True:
        time.sleep(1000)
        os.system("nvidia-smi")

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)


def create_config(dataset_path, target_path, data_path,example_path='config.yml'):
    mkdir(target_path)

    f = open(example_path)
    config = yaml.load(f)

    config['TRAIN_INPAINT_IMAGE_FLIST'] = dataset_path + "/train/images.flist"
    config['TRAIN_INPAINT_LANDMARK_FLIST'] = dataset_path + "/train/landmarks.flist"
    config['TRAIN_MASK_FLIST'] = dataset_path + "/train/masks.flist"

    config['TEST_INPAINT_IMAGE_FLIST'] = dataset_path + "/test/images.flist"
    config['TEST_INPAINT_LANDMARK_FLIST'] = dataset_path + "/test/landmarks.flist"
    config['TEST_MASK_FLIST'] = dataset_path + "/test/masks.flist"

    config['VAL_INPAINT_IMAGE_FLIST'] = dataset_path + "/val/images.flist"
    config['VAL_INPAINT_LANDMARK_FLIST'] = dataset_path + "/val/landmarks.flist"
    config['VAL_MASK_FLIST'] = dataset_path + "/val/masks.flist"
    config['DATA_ROOT'] = data_path

    fr = open(os.path.join(target_path, 'config.yml'), 'w')
    yaml.dump(config, fr)
    fr.close()
    print(target_path)
    print("done")


#python -m torch.distributed.launch --nproc_per_node=2 train.py --model 2 --checkpoint checkpoints/celebahq_styleganbaseae --is_dist

if __name__ == "__main__":
    os.system("pwd ; ls")
    mode = "remote"
    start = time.time()
    
    path_dict = {
        "8613":
            {
                "s3code_project_path": "s3://bucket-8613/chengbin/project/MA-lafin-07-24-17-55",
                "code_path":"/cache/user-job-dir/code",
                "s3data_path":"s3://bucket-8613/chengbin/dataset/celeba-hq/celeba-1024-lafin/",
                "dataset_path": "datasets/celebahqr",
                "config_path": "checkpoints/celebahq-ae-lmfaceidin-256-512-latent/config.yml",
                "checkpoint_name": "celebahq_ae_512",
            },
    }
    
    

    if mode == "remote":
        import moxing as mox
        import os

        path_cfg = "8613"
        
        s3code_path = os.path.join(path_dict[path_cfg]["s3code_project_path"],"code")
        code_path = path_dict[path_cfg]["code_path"]
        

        s3data_path = os.path.join(path_dict[path_cfg]["code_path"],"celebahq_lafin_data")
        data_path = path_dict[path_cfg]["data_path"]

        suffix = time.strftime("%b%d%H%M")
        dataset_path = path_dict[path_cfg]["dataset_path"]
        
        s3savepath = os.path.join(path_dict[path_cfg]["s3code_project_path"], "%s/%s"%("remotecheckpoints",path_dict[path_cfg]["checkpoint_name"]))
        
        #######################################################
        checkpoint_path = "remote_checkpoints/%s-%s"%(path_dict[path_cfg]["checkpoint_name"],suffix)
        #######################################################
        mkdir(checkpoint_path)

        copy_dataset(s3code_path, code_path)
        copy_dataset(s3data_path, data_path)

        sys.path.insert(0, code_path)  # "home/work/user-job-dir/" + leaf folder of src code
        os.chdir(code_path)
        os.system("pwd")

        ######################################################################
        create_config(dataset_path, checkpoint_path,data_path, path_dict[path_cfg]["config_path"])
        ######################################################################

        t = threading.Thread(target=get_checkpoint, args=(checkpoint_path,s3savepath,))
        t.start()

        t = threading.Thread(target=show_nvidia)
        t.start()


        os.system("pwd ; ls")
        os.system("df -h")
        # /cache/user-job-dir/code/ffhq-lafin:
        os.system("pip install -r requirements.txt")

        ## 某些情况
        #os.system("mkdir -p /home/work/.torch/models/")
        #os.system("cp checkpoints/torch/vgg19-dcbb9e9d.pth /home/work/.torch/models/")
        ## 另一些情况
        os.system("mkdir -p /home/work/.cache/torch/checkpoints/")
        os.system("cp checkpoints/torch/vgg19-dcbb9e9d.pth /home/work/.cache/torch/checkpoints/")
        os.system("python -m torch.distributed.launch --nproc_per_node=8 train.py --model 2 --checkpoint %s --data_path %s --is_dist"%(checkpoint_path,data_path))
        #os.system("python train.py --model 2 --checkpoints %s --data_path %s "%(checkpoint_path,dataset_path))
        copy_dataset(checkpoint_path, s3savepath)





