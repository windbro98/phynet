import sys 
sys.path.append("/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet") 
from library import result_visual,mkdir,my_readtxt,my_saveimage
if __name__=='__main__':

    
    # 二、读取的ref或Sam路径
    ref_path = f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_ref_pi_01_prop_pi/2024-02-23-11-09/img_txt_folder/2000_pred.txt'
    # '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_ref_pi_01_prop_pi_mask/2024-02-22-20-26/img_txt_folder/1000_pred.txt'
    sam_path=  f'/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_ref_pi_01_prop_pi/2024-02-23-11-09/img_txt_folder/66500_pred.txt'
    # '/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/result/1536_1536_ref_pi_01_prop_pi_mask/2024-02-22-20-26/img_txt_folder/79500_pred.txt'


    ref = my_readtxt(ref_path)
    sam = my_readtxt(sam_path)   
    diff = (sam - ref)
    save_path = './ref2000_ref66500.png'
    my_saveimage(diff,save_path)  

