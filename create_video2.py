# coding=utf-8
from __future__ import absolute_import, division, print_function
import cv2
import warnings
import numpy as np
warnings.simplefilter("always")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default="checkpoints/obama-stylegan-256-ae-4layers/results/inpaint" ,type=str)
    parser.add_argument('--output_path', default="Obama_stylegan_mouth_inpainting.gif" ,type=str)
    args = parser.parse_args()
    
    root_path = args.root_path
    output_path = args.output_path
  
    width = None
    height = None
    num = 200
    

    fps = 25
    
    import os
   
    path = os.path.join(root_path,"joint")
    files = os.listdir(path)
    ext = os.path.splitext(files[0])[1]
    names = [int(os.path.splitext(file)[0]) for file in files]
    names.sort()
    files = ["%s%s"%(str(name),ext) for name in names]

    vw = VideoWriter(output_path, width, height,fps)

    count = 0
    for file in files:
        print("loading: %s" % os.path.join(path,file))
        frame = cv2.imread(os.path.join(path,file))
        # 写入图像
        vw.write(frame)
        count += 1
        if count == num:
            break 
    # 关闭
    vw.close()


if __name__ == '__main__':
    main()
