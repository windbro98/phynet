'''
print打印网络结构
'''
# import torch
# import torch.nn as nn
# class ConvNet(nn.Module):

#     def __init__(self):

#         super(ConvNet, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, 3, 1, 1),
#             nn.ReLU(),
#             nn.AvgPool2d(2, 2)
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(32 * 7 * 7, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU()
#         )
#         self.out = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         output = self.out(x)
#         return output

# MyConvNet = ConvNet().to('cuda')
# # print(MyConvNet)

'''
torchsummary
'''
from torchsummary import summary
# from torchvision.models import vgg16  # 以 vgg16 为例

# myNet = vgg16().to('cuda')  # 实例化网络，可以换成自己的网络
# summary(myNet, (3, 64, 64))  # 输出网络结构

from unetgood import UnetGenerator
myNet = UnetGenerator().to('cuda')#  # 实例化网络，可以换成自己的网络
# summary(myNet, input_size=(1, 512, 512))  # 输出网络结构
print(myNet)

# from unet import net_model_v1
# myNet = net_model_v1().to('cuda')
# summary(myNet, input_size=(1, 512, 512))  # 输出网络结构



'''
torchinfo
'''

### 打印net_model_v1

# from torchinfo import summary

# from unet import net_model_v1

# myNet = net_model_v1().to('cuda')#  # 实例化网络，可以换成自己的网络
# model_stats = summary(myNet, input_size=(1,1, 512, 512))  # 输出网络结构
# summary_str = str(model_stats)

# net_file = open(f"./net_model_v1.txt","w")
# net_file.write(f'myNet {summary_str}\n')
# net_file.close()

### 打印UnetGenerator

# from torchinfo import summary

# from unetgood import UnetGenerator



# myNet = UnetGenerator().to('cuda')#  # 实例化网络，可以换成自己的网络
# model_stats = summary(myNet, input_size=(1,1, 512, 512))  # 输出网络结构
# summary_str = str(model_stats)

# net_file = open(f"./UnetGenerator.txt","w")
# net_file.write(f'myNet {summary_str}\n')
# net_file.close()

'''
netron
'''
# 针对有网络模型，但还没有训练保存 .pth 文件的情况
# import netron
# import torch.onnx
# from torch.autograd import Variable
# from torchvision.models import resnet18  # 以 resnet18 为例

# from unet import net_model_v1

# x = torch.randn(1, 1, 512, 512)  # 随机生成一个
# myNet = net_model_v1()
# modelData = "./net_model_v1.pth"  # 定义模型数据保存的路径
# torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存

# # myNet = resnet18()  # 实例化 resnet18
# # x = torch.randn(16, 3, 40, 40)  # 随机生成一个输入
# modelData = "./demo.pth"  # 定义模型数据保存的路径
# # modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
# # torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
# netron.start(modelData)  # 输出网络结构