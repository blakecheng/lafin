#coding=utf-8
import cv2
import numpy as np
import  os

'''
for each vedio in vedios
    分解视频成图像
'''

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

SourceImgPath = "/data/qg_data/results"  # 视频读取路径
vedionamelist = os.listdir(SourceImgPath)  # 获得所有视频名字列表

ImgWritePath = "/data/qg_data/Obama_img"  # 图像保存路径
img_end = ".jpg"
img_start = 0

mkdir(ImgWritePath)
print(vedionamelist)

print("From: \n {} \n to: \n {}".format(SourceImgPath,ImgWritePath))

for vedio_path in vedionamelist:
    VedioPath = os.path.join(SourceImgPath, vedio_path) # 获得文件夹下所有文件的路径   读取路径和保存路径
    cap = cv2.VideoCapture(VedioPath)
    while cap.read():
        # get a frame
        ret, frame = cap.read()
        if ret == False:
            print("{} is finished!".format(vedio_path))
            break #读到文件末尾

        # 显示第几帧
        frames_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 显示实时帧数
        FPS = cap.get(cv2.CAP_PROP_FPS)
        # 总帧数
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # show information of frame
        # cv2.putText(frame, "FPS:"+str(FPS), (17, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        # cv2.putText(frame, "NUM OF FRAME:"+str(frames_num), (222, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        # cv2.putText(frame, "TOTAL FRAME:" + str(total_frame), (504, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        # show a frame
        #cv2.imshow("capture", frame)
        # img name
        img_name = str(img_start) + img_end
        img_start = img_start + 1
        # 存储
        if frames_num % 24 == 0:
            cv2.imwrite(os.path.join(ImgWritePath,img_name), frame)
            print("writing {}'s frame {} to {}".format(vedio_path,frames_num,img_name))

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

