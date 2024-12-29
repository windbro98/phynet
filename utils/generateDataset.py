import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
code_path =  '/ailab/user/tangyuhang/ws/phynet-pro'
sys.path.append(code_path)
from utils.generate_mcf_simulate import get_speckle2
# 定义输入和输出文件夹路径
input_folder = '/ailab/user/tangyuhang/dataset/val2017'   # 输入图片的文件夹路径
output_folder = '/ailab/user/tangyuhang/dataset/val2017SamSpeckle'  # 输出图片的文件夹路径
output_folder_gt = '/ailab/user/tangyuhang/dataset/val2017Samgt'  # 输出图片的文件夹路径
output_folder_speckle = '/ailab/user/tangyuhang/dataset/val2017Sam'  # 输出图片的文件夹路径
refPha = np.loadtxt("/ailab/user/tangyuhang/ws/Traindata/simulate_data/2pi/15000/5/None/1920-2560/15000_pha_simulate.txt",dtype=np.float32,delimiter=",")
refAmp = np.loadtxt("/ailab/user/tangyuhang/ws/Traindata/simulate_data/2pi/15000/5/None/1920-2560/15000_amp_simulate.txt",dtype=np.float32,delimiter=",")
# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图片文件
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 处理JPG或PNG格式的图片
        # 获取图片的完整路径
        image_path = os.path.join(input_folder, filename)
        # 打开图像
        image = Image.open(image_path)

        # 获取原始尺寸
        original_width, original_height = image.size

        # 调整图像大小以适应目标尺寸，保持原始纵横比
        if original_width > original_height:
            new_width = 1000
            new_height = int((1000 / original_width) * original_height)
        else:
            new_height = 1000
            new_width = int((1000 / original_height) * original_width)

        # 调整图像大小
        resized_image = image.resize((new_width, new_height))

        # 转换为灰度图像
        gray_image = resized_image.convert('L')

        # 创建目标大小为 1920x2560 的新图像，并填充为 0（黑色）
        target_size = (2560, 1920)
        new_image = Image.new('L', target_size, 0)  # 'L' 表示灰度图像，填充色 0 表示黑色

        # 计算粘贴的位置，使调整大小后的图像位于中央
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2

        # 将调整大小后的灰度图像粘贴到目标图像的中心
        new_image.paste(gray_image, (paste_x, paste_y))

        # 将结果转换为 NumPy 数组
        padded_array = np.array(new_image, dtype=np.float32) / 255.0 *(2*np.pi)  # 归一化到 0-1 之间
        print((padded_array.dtype))
        gt_path = os.path.join(output_folder_gt, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        np.savetxt(gt_path, padded_array,fmt='%.10e',delimiter=",")



        sam = refPha+padded_array

        samPi = np.mod(sam,2*np.pi)

        # 保存 NumPy 数组到 txt 文件
        output_txt_path = os.path.join(output_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        np.savetxt(output_txt_path, samPi,fmt='%.10e',delimiter=",")

        speckle = get_speckle2(0.01,samPi,refAmp)
        # 保存 NumPy 数组到 txt 文件
        speckleoutput_txt_path = os.path.join(output_folder_speckle, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        np.savetxt(speckleoutput_txt_path, speckle,fmt='%.10e',delimiter=",")

plt.figure()
plt.imshow(padded_array)
plt.savefig('/ailab/user/tangyuhang/dataset/gt.jpg')

plt.figure()
plt.imshow(samPi)
plt.savefig('/ailab/user/tangyuhang/dataset/phaDistortion.jpg')

plt.figure()
plt.imshow(speckle)
plt.savefig('/ailab/user/tangyuhang/dataset/speckle.jpg')


print("图片处理完成并保存到输出文件夹。")