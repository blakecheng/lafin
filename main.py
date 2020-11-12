import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.lafin import Lafin


def main(mode=None):
    r"""starts the model
    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """
    os.system("pwd")
    config = load_config(mode)
    print(config.GPU)
    
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = Lafin(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        with torch.autograd.set_detect_anomaly(True):
            model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()
        
    elif config.MODE == 3:
        print('\nstart training finetune...\n')
        model.train_finetune()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config
    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], help='1: landmark prediction model, 2: inpaint model, 3: joint model')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument("--local_rank", type=int, default=0) 
    parser.add_argument("--is_dist",action='store_true')
    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--landmark', type=str, help='path to the landmarks directory or a landmark file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()

    
    
    torch.cuda.set_device(args.local_rank)
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    if args.data_path is not None:
        config.DATA_ROOT = args.data_path
    else:
        config.DATA_ROOT = config.DATA_ROOT
    
    if args.is_dist:
        torch.distributed.init_process_group(backend="nccl")
        config.DISTRIBUTED = True
        torch.cuda.set_device(args.local_rank)
    config.LocalRank = args.local_rank
        
    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3

        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.landmark is not None:
            config.TEST_INPAINT_LANDMARK_FLIST = args.landmark

        if args.output is not None:
            config.RESULTS = args.output

        

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main()
