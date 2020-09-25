import os
import argparse

## 传code文件夹
## 传某个文件
os.system("python3 uploader_yundao.py \
--local_folder_absolute_path=/home/cb/project/lafin/code \
--app_token=63dc27e2-740f-46a5-b58b-1c742aa98d7c \
--vendor=HEC \
--region=cn-north-1 \
--bucket_name=bucket-8613 \
--bucket_path=chengbin/project/MA-lafin-07-24-17-55")