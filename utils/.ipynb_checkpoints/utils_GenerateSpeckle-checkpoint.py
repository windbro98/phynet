import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet")

import os
from os.path import join, getsize
import scipy.io as scio
def my_rename(path):
    for root, dirs, files in os.walk(path):
        
        print(f'in {root}')
        
        for file in (files):
            file_old_path = join(root,file)
            print(file)
            if file == 'phase_cal_sam.mat':
                
                os.rename(file_old_path,join(root,'phase_diff.mat'))
                
            elif file == 'phase_diff.mat':
                
                pass
            
            elif file == 'phase_diff.mat':
                pass

var_dict={}
import os

# 定义包含.txt文件的三个文件夹的路径
folders = ['path/to/folder1', 'path/to/folder2', 'path/to/folder3']

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


for root, dirs, files in os.walk('/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/traindata/raw'):
    
    print(f'in {root}')
    
    for file in (files):
        
        pha = scio.loadmat(join(root,file))
        print(pha)
        # print(type(pha),pha.shape,pha.max,pha.min)
        
            
    

    