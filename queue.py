import os
from datetime import datetime

import subprocess
import sys
import time
import yaml

import threading
import inspect
import ctypes


# import moxing as mox
def copy_file(obs_path, cache_path, mode="remote"):
    if not os.path.exists(os.path.dirname(cache_path)): os.makedirs(os.path.dirname(cache_path))
    print('start copy {} to {}: {}'.format(obs_path, cache_path, datetime.now().strftime("%m-%d-%H-%M-%S")))
    if mode == "local":
        command("cp {} {}".format(obs_path, cache_path))
    else:
        mox.file.copy(obs_path, cache_path)

    print('end copy {} to cache: {}'.format(obs_path, datetime.now().strftime("%m-%d-%H-%M-%S")))


def copy_dataset(obs_path, cache_path, mode="remote"):
    if not os.path.exists(cache_path): os.makedirs(cache_path)
    print('start copy {} to {}: {}'.format(obs_path, cache_path, datetime.now().strftime("%m-%d-%H-%M-%S")))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if mode == "local":
        command("cp {} {}".format(obs_path, cache_path))
    else:
        mox.file.copy_parallel(obs_path, cache_path)
    print('end copy {} to cache: {}'.format(obs_path, datetime.now().strftime("%m-%d-%H-%M-%S")))


def command(cmd="ls"):
    d = subprocess.getstatusoutput(str(cmd))
    if d[0] == 0:
        print("Command success: {}".format(cmd))
        print("Output: \n {}".format(d[1]))
    else:
        print("Command fail: {}".format(cmd))
        print("Error message: \n {}".format(d[1]))
    return d[0]


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def create_task(task_list):
    f = open(project_yml)
    config = yaml.load(f)
    f.close()
    task_dict = config['task']
    for key, value in task_dict.items():
        if value["is_excuted"] == False:
            try:
                print("Adding task : {} {} ".format(key, value))
                t = threading.Thread(target=os.system, args=(value['command'],), name=key)
                t.start()
                task_list.append(t)
                print("Adding finish , total task : {} ".format(len(task_list)))
                config['task'][key]["is_excuted"] = True
            except Exception as e:
                print(e)

    return task_list


os.system("pwd ; ls")

# s3code_path = "s3://bucket-333/chengbin/code/lafin/Lafin/code"
# code_path = "/home/ma-user/work/code"

s3code_path = "/data/chengbin/code/lafin"
code_path = "/data/chengbin/code/lafin/workspace"

checkpoint_path = "checkpoints/celeba1024-all-512"
project_yml = "project.yml"

copy_file(s3code_path + "/" + project_yml, code_path + "/" + project_yml, "local")

sys.path.insert(0, code_path)  # "home/work/user-job-dir/" + leaf folder of src code
os.chdir(code_path)
command("pwd")

init = time.time()
start = time.time()

task_list = []
while True:
    time.sleep(1)
    end = time.time()

    if (int(end - start)) % 4 == 0:
        f = open(project_yml)
        config = yaml.load(f)
        f.close()
        print("Load Status !")
        copy_file(s3code_path + "/" + project_yml, code_path + "/" + project_yml, "local")
        if config["is_excuted"] == False:
            print("Prepare runing, Total Time:{:.4f} , Excuting.....".format(end - init))
            try:
                print("Project update: \n", config)
                print("creating task ...")
                task_list = create_task(task_list)
            except Exception as e:
                print(e)
            finally:
                config["is_excuted"] = True
                fr = open(project_yml, 'w')
                yaml.dump(config, fr)
                fr.close()
                copy_file(code_path + "/" + project_yml, s3code_path + "/" + project_yml, "local")

        print("Total Time:{:.4f}".format(end - init))
        start = time.time()
