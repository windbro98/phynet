'''
训练过程速度提高

'''

import os
import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet") 
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from library import (my_readtxt,mkdir,visual_data,prop,my_saveimage,my_savetxt)

from config.parameter import Parameter,import_class
from trainer.trainer import train_epoch

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import argparse

def cropImage(img,image_width = 1920,image_height = 2560,crop_width = 1920,crop_height = 2048):
    
    # 图像尺寸 image_width = 1920,image_height = 2560
    
    
    # 截取矩形的尺寸 crop_width = 1920,crop_height = 2048
    
    
    # 计算截取矩形的左上角坐标
    x_coordinate = (image_width - crop_width) // 2
    y_coordinate = (image_height - crop_height) // 2
    # 截取图像
    cropped_img = img[x_coordinate:x_coordinate + crop_width,y_coordinate:y_coordinate + crop_height]
    
    return cropped_img

def getCropImage(file_path,crop_width = 1920,crop_height = 2048):
    # 读取并打印文件内容
    amp = np.loadtxt(file_path,dtype=np.float32,delimiter=',')
    print(f"内容来自文件 {filename}:")
    print(f'shape of data:{amp.shape}')
    print("---------------------")
    
    # 获取对应的相位路径
    pha_path = file_path.replace('amp', 'pha').replace('_prop001', '')
    # 打印对应的相位路径
    
    pha = np.loadtxt(pha_path,dtype=np.float32,delimiter=',')
    print(f"内容来自文件 {pha_path}:")
    print(f'shape of data:{pha.shape}')
    print("---------------------")

    # 截取图像
    cropped_amp = cropImage(amp,crop_width = crop_width,crop_height = crop_height)
    
    # 截取图像
    cropped_pha = cropImage(pha,crop_width = crop_width,crop_height = crop_height)
    

    
    return cropped_amp,cropped_pha    

class mydataset(torch.utils.data.Dataset):
    # 从txt文件中读矩阵
    def __init__(self,data):

        self.data = data

        
    def __getitem__(self,idx):

        # return x,y
        return self.data,self.data #使数据类型和模型参数类型一样

    def __len__(self):

        return 1    

# def train_epoch(train_dataloader,net,loss_mse,optimizer,para,current_epoch,cropped_pha,best_loss):
    
#     for batch,(x,y) in (enumerate(train_dataloader)):
        
#         optimizer.zero_grad()
#         # forward proapation
#         pred_y = net(x) 
        
#         flattened_pred_y = pred_y[0, 0, :, :]      
#         measured_y = prop(flattened_pred_y,dist=para.dist)
#         loss_mse_value = loss_mse(y.float(),measured_y.float())
#         loss_value =  loss_mse_value

#         # backward proapation
#         loss_value.backward()
        
#         optimizer.step()
        
#         return loss_value,flattened_pred_y
        
def train_epoch(train_dataloader,net,loss_mse,optimizer):
    for (x,y) in (train_dataloader):
        flattened_x = x[0, 0, :, :] 
        optimizer.zero_grad()
        # forward proapation
        pred_y = net(x) 
        
        flattened_pred_y = pred_y[0, 0, :, :]      
        measured_y = prop(flattened_pred_y,dist=para.dist)
        loss_mse_value = loss_mse(flattened_x.float(),measured_y.float())
        loss_value =  loss_mse_value

        # backward proapation
        loss_value.backward()
        
        optimizer.step()   

        return flattened_pred_y,measured_y,loss_value
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--opt', type=str, default='/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet/option/baseline2.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    para = Parameter(args.opt)
    
    

    localtime = time.strftime("%Y-%m-%d-%H-%M", time.localtime())    
 
    result_folder = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/{para.exp_name}/{localtime}'

    tb_folder = f'{result_folder}/tb_folder'
    weight_folder = f"{result_folder}/weight_folder"
    img_txt_folder = f'{result_folder}/img_txt_folder'
    
    # 最好的loss初始化为无穷大
    best_loss = float('inf')

    # 随机种子和实验设备
    torch.manual_seed(para.seed)
    
    # hypara
    batch_size = para.batch_size
    lr = para.lr
    epochs = para.epochs
    print(batch_size,lr,epochs)
    # 初始化存储所有图片loss列表的列表
    all_experiment_losses = []
    
    device = torch.device(para.device) #分布式训练时用默认    
    
    # 定义文件夹路径
    folder_paths = para.folder_paths
    # 遍历每个文件夹
    for folder in folder_paths:
        # 检查文件夹是否存在
        if os.path.exists(folder):
            # 遍历文件夹内的所有文件
            for filename in os.listdir(folder):
                # 检查文件是否为.txt文件
                if filename.endswith('.txt'):
                    # 拼接完整文件路径
                    file_path = os.path.join(folder, filename)
                    cropped_amp,cropped_pha = getCropImage(file_path,crop_width = para.image_height,crop_height = para.image_width)
                    
                    # 保存图像
                    mkdir(folder.replace('amp','amp_cropped'))
                    cropped_amp_imgpath = file_path.replace('amp','amp_cropped').replace('txt','png')
                    cropped_amp_txtpath = file_path.replace('amp','amp_cropped')#.replace('png','txt')
                    my_saveimage(cropped_amp,cropped_amp_imgpath)
                    my_savetxt(cropped_amp,cropped_amp_txtpath)
                    print(cropped_amp_imgpath)
                    print(cropped_amp_txtpath)
                    
                    # 保存图像
                    mkdir(folder.replace('amp','pha_cropped'))
                    cropped_pha_imgpath = file_path.replace('amp','pha_cropped').replace('txt','png')
                    cropped_pha_txtpath = file_path.replace('amp','pha_cropped')#.replace('png','txt')
                    my_saveimage(cropped_pha,cropped_pha_imgpath)
                    my_savetxt(cropped_pha,cropped_pha_txtpath)
                    print(cropped_pha_imgpath)
                    print(cropped_pha_txtpath)
                    print("1111111111111111111111111111end111111111111111111111111111")
 

                    
        else:
            print(f"文件夹 {folder} 不存在。")

   




        




