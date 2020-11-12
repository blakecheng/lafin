import os
import sys
## 传code文件夹
## 传某个文件

if len(sys.argv)==2:
    print( sys.argv)
    source = sys.argv[1]
    os.system("python3 uploader_yundao.py \
    --local_folder_absolute_path=/home/cb/project/lafin/code/%s \
    --app_token=63dc27e2-740f-46a5-b58b-1c742aa98d7c \
    --vendor=HEC \
    --region=cn-north-4 \
    --bucket_name=bucket-333 \
    --bucket_path=chengbin/project/MA-lafin-07-24-17-55/code"%(source))
elif len(sys.argv)==3:
    print(sys.argv)
    source = sys.argv[1]
    target = sys.argv[2]
    os.system("python3 uploader_yundao.py \
    --local_folder_absolute_path=%s \
    --app_token=63dc27e2-740f-46a5-b58b-1c742aa98d7c \
    --vendor=HEC \
    --region=cn-north-4 \
    --bucket_name=bucket-333 \
    --bucket_path=%s"%(source,target))
else:
    os.system("python3 uploader_yundao.py \
    --local_folder_absolute_path=/home/cb/project/lafin/code \
    --app_token=63dc27e2-740f-46a5-b58b-1c742aa98d7c \
    --vendor=HEC \
    --region=cn-north-4 \
    --bucket_name=bucket-333 \
    --bucket_path=chengbin/project/MA-lafin-07-24-17-55")
