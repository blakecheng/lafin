# coding=utf-8
from __future__ import absolute_import, division, print_function
import cv2
import warnings
import numpy as np
warnings.simplefilter("always")


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.avi'):  # 保证文件名的后缀是.avi
            name += '.avi'
            warnings.warn('video name should ends with ".avi"')
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width        # 宽
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 如果是avi视频，编码需要为MJPG
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()


def main():
    width = 1024
    height = 256
    
    num = 200
    # root_path = "/data/chengbin/code/lafin/checkpoints/Obama_face_10_train/Obama_face/results/inpaint"
    root_path = "/home/public/cb/code/lafin_bak/checkpoints/celebahq-stylegan-256-ae-4layers/results/inpaint"
    output_path = 'Obama_inpainting.avi'
    fps = 1 
    
    import os
   
    path = os.path.join(root_path,"joint")
    files = os.listdir(path)
    ext = os.path.splitext(files[0])[1]
    names = [int(os.path.splitext(file)[0]) for file in files]
    names.sort()
    files = ["%s%s"%(str(name),ext) for name in names]

    vw = VideoWriter(os.path.join(root_path,output_path), width, height,fps)

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
