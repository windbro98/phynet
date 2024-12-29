import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet")

import os
from os.path import join, getsize
import scipy.io as scio
import numpy as np
from library import prop,my_saveimage,mkdir,my_savetxt
import matplotlib.pyplot as plt
import torch
def my_rename():
    var_dict={}

    # root = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/raw_txt'
    # 定义包含.txt文件的三个文件夹的路径
    folders = ['/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/raw_txt/blood', 
            '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/raw_txt/colon1',
            '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/raw_txt/colon2']

    # 定义要重命名的文件及其新名称
    rename_rules = {
        'pha_prop_cal.txt': 'phase_ref.txt',
        'phase_cal_sam.txt': 'phase_diff.txt',
        # 'phase_sam.txt': 'phase_sam.txt' # 这行是不必要的，因为文件名未改变
    }

    # 遍历每个文件夹
    for folder in folders:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder):
            old_file_path = os.path.join(folder, filename)
            
            # 检查文件是否在重命名规则中
            if filename in rename_rules:
                new_file_path = os.path.join(folder, rename_rules[filename])
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f'Renamed "{old_file_path}" to "{new_file_path}"')


def generatespeckle(root,save_root,scale='2pi'):
    for dir in os.listdir(root):
        
        print(f'in {dir}')
        
        for file in  os.listdir(join(root,dir)):
            
            pha = np.loadtxt(join(root,dir,file),dtype=np.float32,delimiter=',')
            
            if scale == '2pi':
                pha = pha           
            elif scale == 'pi':
                pha = (pha+np.pi)/2
                
            print(pha.max(),pha.min())
            
            pha_savedir = join(save_root,f'{scale}','pha',dir)
            mkdir(pha_savedir)
            my_saveimage(pha,join(pha_savedir,f'{file[:-4]}.png'))
            my_savetxt(pha,join(pha_savedir,f'{file[:-4]}.txt'))
            
            pha = torch.tensor(pha)#.to('cuda:0')
        #     plt.imshow(pha,cmap='viridis')
        #     print(pha.dtype,pha.shape,pha.max(),pha.min(),pha.sum())
            
            amp = prop(pha,dist=0.01,device='cpu')
            
            amp_savedir = join(save_root,f'{scale}','amp',dir)
            mkdir(amp_savedir)
            my_saveimage(amp,join(amp_savedir,f'{file[:-4]}_prop001.png'))
            my_savetxt(amp,join(amp_savedir,f'{file[:-4]}_prop001.txt'))
root = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/raw_txt'
save_root = '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata'    
scale = 'pi'    
generatespeckle(root,save_root,scale=scale)            
    

    