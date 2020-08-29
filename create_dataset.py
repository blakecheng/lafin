import os
import subprocess
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm 
import yaml
import argparse
import face_alignment 

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

def command(cmd="ls"):
    d=subprocess.getstatusoutput(str(cmd))
    if d[0] == 0:
        print("Command success: {}".format(cmd))
        print("Output: \n {}".format(d[1]))
    else:
        print("Command fail: {}".format(cmd))
        print("Error message: \n {}".format(d[1]))  

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
    
    
def to_plt(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     

def create_mask(path="datasets/debug/", img_dir = 'datasets/debug/images/',
                mask_dir='datasets/debug/masks/', 
                landmark_dir = 'datasets/debug/landmasks/',
                img_flist = '/datasets/debug/images.flist',
                mask_flist = "/datasets/debug/masks.flist",
                landmark_flist = '/datasets/debug/landmarks.flist',
                img_num = None):
    
    mkdir(img_dir)
    mkdir(mask_dir)
    mkdir(landmark_dir)
    img_list = os.listdir(path)
    
    if img_num is not None:
        img_list = np.random.choice(img_list,img_num)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    for file_name in tqdm(img_list):
       
        file_name_no_ext = os.path.splitext(file_name)[0]
        if os.path.exists(os.path.join(mask_dir,file_name_no_ext+'.png')):
            continue

        
        input_img = cv2.imread(os.path.join(path,file_name))
        
        try:
            if input_img==None:
                print("Error occured when loading: \n %s" % os.path.join(path,file_name))
                continue
        except:
            pass

        try:
            preds = fa.get_landmarks(input_img)
        except:
            continue

        if preds==None:
            print("Can't detect landmark from pic showed blow")
#             tmp_path = path+"_no_landmark"
#             mkdir(tmp_path)
#             command("cp {}/{} {}/{}".format(path,file_name,tmp_path,file_name))
            #SHOW([to_plt(input_img)])
            
            continue
           
        
        
        x, y, w, h = cv2.boundingRect(np.concatenate((np.array(preds[0][3:14,:]),np.array(preds[0][48:67,:])),axis=0))
        mask_img = cv2.rectangle(input_img.copy(), (x, y), (x + w, y + h), (255, 255, 255), -1)
        mask = cv2.rectangle(np.zeros(input_img.shape,input_img.dtype), (x, y), (x + w, y + h), (255, 255, 255), -1)
       
        
        
        if list(img_list).index(file_name)<5:
            landmark_cord = preds[0]
            for i in range(landmark_cord.shape[0]):
                center = (int(landmark_cord[i,0]),int(landmark_cord[i,1]))
                cv2.putText(mask_img, str(i), center, cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
                cv2.circle(mask_img,center, radius=5,color=(0, 0, 255), thickness=-1)

            #SHOW([to_plt(input_img),to_plt(mask_img),to_plt(mask)])
            
        cv2.imwrite(os.path.join(img_dir , file_name),input_img)
        cv2.imwrite(os.path.join(mask_dir, file_name_no_ext+'.png'),mask)
        
    path_to_mask = mask_dir
    path_to_mask_flist= mask_flist
    create_mask_flist_cmd = "python ./scripts/flist.py --path {} --output {}".format(path_to_mask,path_to_mask_flist)
    command(create_mask_flist_cmd)
    
    create_image_flist_cmd = "python ./scripts/flist.py --path {} --output {}".format(img_dir,img_flist)
    command(create_image_flist_cmd)
    
#     create_landmark_cmd = "python3 ./scripts/preprocess_landmark.py --path {} --output {}".format(img_dir,landmark_dir)
#     command(create_landmark_cmd)
    
    filenames = os.listdir(img_dir)
    for filename in tqdm(filenames):
        if filename[-3:] != 'png' and filename[-3:] != 'jpg':
            continue
            
        if os.path.exists(os.path.join(landmark_dir,filename[:-4]+'.txt')):
            with open(os.path.join(landmark_dir,filename[:-4]+'.txt'),'r') as f:
                data = f.readlines()
                landmarks= np.genfromtxt(data)
                if landmarks.shape[0]!=136:
                    print("Correct file: %s"%(filename[:-4]+'.txt'))
                else:
                    continue
            
        with open(os.path.join(landmark_dir,filename[:-4]+'.txt'), 'w') as f:
            img = io.imread(os.path.join(img_dir,filename))
            l_pos = fa.get_landmarks(img)
            
            if l_pos==None:
                print("Can't detect landmark from pic showed blow")
#                 tmp_path = path+"_no_landmark"
#                 mkdir(tmp_path)
#                 command("mv {}/{} {}/{}".format(img_dir,filename,tmp_path,filename))
#                SHOW([to_plt(img)])
                continue
            
            for i in range(68):
                try:
                    f.write(str(l_pos[0][i,0])+' '+str(l_pos[0][i,1])+' ')
                except:
                    print("Erro when processing {}".format(filename))
                   
            f.write('\n')

    
    create_landmark_flist_cmd = "python3 ./scripts/flist.py --path {} --output {}".format(landmark_dir,landmark_flist)
    command(create_landmark_flist_cmd)
    print("images : {} , masks : {} , landmarks : {} ".format(len(os.listdir(img_dir)),len(os.listdir(mask_dir)),len(os.listdir(landmark_dir))))


def create_dataset(nums,img_dir,mask_dir,landmark_dir,output_path,mode="relative"):
    files = os.listdir(img_dir)
    ext = os.path.splitext(files[0])[1]
    files = [os.path.splitext(i)[0] for i in files]
    file_num = len(files)
    file_nums = [int(i*file_num) for i in nums]
    print("file nums:", file_nums)
    
    trainfiles = files[:file_nums[0]]
    valfiles = files[file_nums[0]+1: file_nums[0]+file_nums[1]]
    testfiles = files[file_nums[0]+file_nums[1]+1:]
    
    filedict = {
        "train": trainfiles,
        "val" : valfiles,
        "test": testfiles
    }
    
    
    mkdir(output_path)
    mkdir(os.path.join(output_path,"train"))
    mkdir(os.path.join(output_path,"val"))
    mkdir(os.path.join(output_path,"test"))
    
    
    
    def create_flist(img_dir, mask_dir, landmark_dir, input_files,output_path,ext = ".jpg",mode="relative"):
        images = []
        masks = []
        landmarks =[]
        for file in input_files:
            if mode == "relative":
                images.append(file + ext)
                masks.append(file +".png")
                landmarks.append( file +".txt")
            else:
                images.append(os.path.join(img_dir, file)+ ext)
                masks.append(os.path.join(mask_dir, file)+".png")
                landmarks.append(os.path.join(landmark_dir, file)+".txt")
        images = sorted(images)
        masks = sorted(masks)
        landmarks = sorted(landmarks)

        np.savetxt(os.path.join(output_path,"images.flist"), images, fmt='%s')
        np.savetxt(os.path.join(output_path,"masks.flist"), masks, fmt='%s')
        np.savetxt(os.path.join(output_path,"landmarks.flist"), landmarks, fmt='%s')
        
    create_flist(img_dir,mask_dir,landmark_dir,filedict['train'],os.path.join(output_path,"train"),ext=ext,mode=mode)
    create_flist(img_dir,mask_dir,landmark_dir,filedict['val'],os.path.join(output_path,"val"),ext=ext,mode=mode)
    create_flist(img_dir,mask_dir,landmark_dir,filedict['test'],os.path.join(output_path,"test"),ext=ext,mode=mode)
    
def create_config(dataset_path,target_path,example_path='config.yml'):
   
    mkdir(target_path)

    f = open(example_path) 
    config = yaml.load(f) 

    config['TRAIN_INPAINT_IMAGE_FLIST'] = dataset_path+"/train/images.flist" 
    config['TRAIN_INPAINT_LANDMARK_FLIST'] = dataset_path+"/train/landmarks.flist" 
    config['TRAIN_MASK_FLIST'] = dataset_path+"/train/masks.flist"

    config['TEST_INPAINT_IMAGE_FLIST'] = dataset_path+"/test/images.flist" 
    config['TEST_INPAINT_LANDMARK_FLIST'] = dataset_path+"/test/landmarks.flist" 
    config['TEST_MASK_FLIST'] = dataset_path+"/test/masks.flist"

    config['VAL_INPAINT_IMAGE_FLIST'] = dataset_path+"/val/images.flist" 
    config['VAL_INPAINT_LANDMARK_FLIST'] = dataset_path+"/val/landmarks.flist" 
    config['VAL_MASK_FLIST'] = dataset_path+"/val/masks.flist"

    fr = open(os.path.join(target_path,'config.yml'), 'w')
    yaml.dump(config, fr)
    fr.close()
    print(target_path)
    print("done")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pic', default='/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin-debug/' ,type=str, help='path to the dataset')
    parser.add_argument('--dataset', default="datasets/celeba1024-debug" ,type=str, help='path to the file list')
    parser.add_argument('--checkpoint', default="checkpoints/celeba1024-debug" ,type=str, help='path to the file list')
    parser.add_argument('--config',default='config.yml',type = str, help = 'path to the config')
    parser.add_argument('--from_raw',type = bool)
    parser.add_argument("--raw",default='/data/chengbin/dataset/celebA/HQ_zip/celeba-hq/celeba-1024-lafin/' ,type=str, help='path to the raw')
    parser.add_argument('--rate',default=[0.9,0.05,0.05],nargs='+', type=float )
    args = parser.parse_args()
    
    if args.from_raw == True:
        img_dir =  os.path.join(args.pic,"images")
        img_flist = os.path.join(args.pic,'images.flist')
        mask_dir= os.path.join(args.pic,"masks")
        mask_flist = os.path.join(args.pic,"masks.flist")
        landmark_dir = os.path.join(args.pic,"landmarks")
        landmark_flist = os.path.join(args.pic,"landmarks.flist")
        
        create_mask(args.raw,img_dir,mask_dir,landmark_dir,img_flist,mask_flist,landmark_flist)
    
    
    # nums=args.rate
    # pic_path = args.pic
    # dataset_path = args.dataset
    # checkpoint_path = args.checkpoint

    # img_dir= os.path.join(pic_path,"images")
    # mask_dir= os.path.join(pic_path,"masks")
    # landmark_dir = os.path.join(pic_path,"landmarks")

    # create_dataset(nums,img_dir,mask_dir,landmark_dir,dataset_path)
    # create_config(dataset_path,checkpoint_path,example_path=args.config)
    # print("dataset locate：%s ,checkpoint locate : %s "%(dataset_path,checkpoint_path))
    
    

  
