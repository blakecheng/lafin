import os
from datetime import datetime
import time


import sys


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


if __name__ == "__main__":
    os.system("pwd ; ls")
    mode = "local"

    if mode == "remote":
        import moxing as mox


        s3code_path = "s3://bucket-333/chengbin/code/lafin/Lafin/code"
        code_path = "/cache/user-job-dir/code"

        s3data_path = "s3://bucket-333/chengbin/dataset/celeba-hq/celeba-1024-lafin"


        data_path = "/cache/user-job-dir/code/celeba-1024-lafin"

        suffix = time.strftime("%b%d%H%M")
        dataset_path = "datasets/celeba1024-all-%s"%(suffix)
        checkpoint_path = "remote_checkpoints/celeba1024-all-%s"%(suffix)


        sys.path.insert(0, code_path)  # "home/work/user-job-dir/" + leaf folder of src code
        os.chdir(code_path)
        os.system("pwd")

        copy_dataset(s3code_path, code_path)
        copy_dataset(s3data_path,data_path)

        os.system("pwd ; ls")
        os.system("df -h")



        os.system("pip install -r requirements.txt")
        os.system("mkdir -p /home/work/.torch/models/")
        os.system("cp checkpoints/torch/vgg19-dcbb9e9d.pth /home/work/.torch/models/")
        os.system("python create_dataset.py --pic %s --dataset %s --checkpoint %s"%(
            data_path,dataset_path,checkpoint_path
        ))
        os.system("python train.py --model 2 --checkpoints %s"%(checkpoint_path))
        copy_dataset(checkpoint_path, s3code_path+"/" + checkpoint_path)

    elif mode == "local":
        suffix = time.strftime("%b%d%H%M")
        data_path = '/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin'
        dataset_path = 'datasets/celeba1024-all-%s' % suffix
        checkpoint_path = 'checkpoints/celeba1024-all-%s' % suffix
        print("start locally!")
        os.system("python create_dataset.py --pic {} --dataset {} --checkpoint {}".format(data_path,dataset_path,checkpoint_path))
        os.system("python train.py --model 2 --checkpoints {}".format(checkpoint_path))


