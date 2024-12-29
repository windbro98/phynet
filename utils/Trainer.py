# '''
# 训练过程速度提高

# '''

# import os

# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# import math
# import time
# import numpy as np
# import matplotlib.pyplot as plt

# from library import (my_readtxt,mkdir,visual_data,prop,my_saveimage,my_savetxt)
# # from unet import net_model_v1
# from unetgood import UnetGenerator
# from baseunet import UnetGeneratorDouble,UNet

# from loss import TVLoss
# from dataset import measured_y_txt_dataset256,measured_y_txt_dataset256_fast

# import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter

# from tqdm import tqdm
# import argparse
# from config.parameter import Parameter,import_class


# def train_loop(para,train_dataloader,net,loss_mse,optimizer):

#     for batch,(x,y) in (enumerate(train_dataloader)):
        
#         optimizer.zero_grad()

#         pred_y = net(x)        
#         measured_y = prop(pred_y[0, 0, :, :],dist=para.dist)

#         loss_mse_value = loss_mse(y.float(),measured_y.float())
#         loss_value =  loss_mse_value

#         # backward proapation

#         loss_value.backward()
#         optimizer.step()
import numpy as np
import matplotlib.pyplot as plt

# 假设每次实验包含100次循环，每次循环都会产生一个loss值
num_experiments = 5  # 你想运行的实验次数
num_iterations = 100  # 每个实验的循环次数

# 初始化存储所有实验loss列表的列表
all_experiment_losses = []

# 运行每个实验并记录loss
for _ in range(num_experiments):
    # 初始化实验的loss列表
    experiment_losses = []
    
    # 运行100次循环
    for _ in range(num_iterations):
        # 这里应该是你的实际loss计算
        # loss = ...
        loss = np.random.rand()  # 模拟一个loss值
        experiment_losses.append(loss)
    
    # 将当前实验的loss列表添加到总列表中
    all_experiment_losses.append(experiment_losses)

# 计算每个实验的平均loss
average_losses = [np.mean(all_experiment_losses, axis=0)]

# 打印每个实验的平均loss
print(average_losses)

# 绘制loss曲线图
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# 遍历每个实验的loss列表
for i, losses in enumerate(all_experiment_losses):
    axs[i // 3, i % 3].plot(range(1, num_iterations + 1), losses, label=f'Experiment {i+1} Loss')

    # 计算并绘制平均loss
    average_loss = np.mean(all_experiment_losses[i])
    axs[i // 3, i % 3].plot(range(1, num_iterations + 1), [average_loss] * num_iterations, label='Average Loss', linestyle='--')

    # 设置图表标题和标签
    axs[i // 3, i % 3].set_title(f'Experiment {i+1} Loss Over迭代ations')
    axs[i // 3, i % 3].set_xlabel('迭代ation')
    axs[i // 3, i % 3].set_ylabel('Loss')
    axs[i // 3, i % 3].legend()
    axs[i // 3, i % 3].grid(True)

# 调整子图间距
plt.tight_layout()

# 保存图表为PNG图片
plt.savefig('loss_over_iterations_multiple_experiments.png', format='png')

# 显示图表
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # 假设每次实验包含100次循环，每次循环都会产生一个loss值
# num_experiments = 5  # 你想运行的实验次数
# num_iterations = 100  # 每个实验的循环次数

# # 初始化存储所有实验loss列表的列表
# all_experiment_losses = []

# # 运行每个实验并记录loss
# for _ in range(num_experiments):
#     # 初始化实验的loss列表
#     experiment_losses = []
    
#     # 运行100次循环
#     for _ in range(num_iterations):
#         # 这里应该是你的实际loss计算
#         # loss = ...
#         loss = np.random.rand()  # 模拟一个loss值
#         experiment_losses.append(loss)
    
#     # 将当前实验的loss列表添加到总列表中
#     all_experiment_losses.append(experiment_losses)

# # 计算每个实验的平均loss
# average_losses = [np.mean(all_experiment_losses, axis=0)]

# # 打印每个实验的平均loss
# print(average_losses)

# # 绘制loss曲线图
# plt.figure(figsize=(10, 6))

# # # 绘制每个实验的loss列表
# # for i, losses in enumerate(all_experiment_losses):
# #     plt.plot(range(1, num_iterations + 1), losses, label=f'Experiment {i+1} Loss')

# # 绘制平均loss
# plt.plot(range(1, num_iterations + 1), average_losses[0], label='Average Loss', linestyle='--')

# # 设置图表标题和标签
# plt.title('Loss Over迭代ations for Multiple Experiments')
# plt.xlabel('迭代ation')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# # 保存图表为PNG图片
# plt.savefig('loss_over_iterations.png', format='png')

