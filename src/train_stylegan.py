from retry.api import retry_call
from tqdm import tqdm
from stylegan2_pytorch.stylegan2_lm_pytorch import Trainer, NanException
from datetime import datetime

def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 128,
    network_capacity = 16,
    transparent = False,
    batch_size = 3,
    gradient_accumulate_every = 5,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    num_workers =  None,
    save_every = 1000,
    generate = False,
    generate_interpolation = False,
    save_frames = False,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    fp16 = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.,
    dataset_aug_prob = 0.,
):
    model = Trainer(
        name,        
        results_dir,
        models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        transparent = transparent,
        lr = learning_rate,
        num_workers = num_workers,
        save_every = save_every,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        dataset_aug_prob = dataset_aug_prob
    )

    if not new:
        model.load(load_from)
    else:
        model.clear()

    if generate:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        model.evaluate(samples_name, num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        model.generate_interpolation(samples_name, num_image_tiles, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    model.set_data_src(data)

    for _ in tqdm(range(num_train_steps - model.steps), mininterval=10., desc=f'{name}<{data}>'):
        retry_call(model.train, tries=3, exceptions=NanException)
        if _ % 50 == 0:
            model.print_log()
            
if __name__ == "__main__":
    import os 
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/data/chengbin/dataset/FFHQ/ffhq-lafin/images/')
    parser.add_argument('--results_dir', type=str, default='./result')
    parser.add_argument('--models_dir', type=str, default='./models')
    parser.add_argument('--name', type=str, default='FFHQ-lm')
    parser.add_argument('--new', type=bool, default=False)
    parser.add_argument('--load_from',type=int,default=-1)
    parser.add_argument('--image_size',type=int,default= 256)
    parser.add_argument('--batch_size',type=int,default= 3)
    parser.add_argument("--local_rank", type=int, default=0) 
    args = parser.parse_args()
    
    
    
    train_from_folder(
    data = args.data,
    results_dir = args.results_dir,
    models_dir = args.models_dir,
    name = args.name,
    new = args.new,
    load_from = args.load_from,
    image_size = args.image_size,
    network_capacity = 16,
    transparent = False,
    batch_size = args.batch_size,
    gradient_accumulate_every = 5,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    num_workers =  None,
    save_every = 100,
    generate = False,
    generate_interpolation = False,
    save_frames = True,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    fp16 = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.,
    dataset_aug_prob = 0.,
)