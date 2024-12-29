import numpy as np
import matplotlib.pyplot as plt


ref_pred_path = "/ailab/user/tangyuhang/ws/Resultimulate/DifferentEncoderDecoder/simulate_EncoderDecoder/U_Net/4/Simulate/2024-10-15-16-40/ref/img_txt_folder/1800_PredPhawrap2pi.txt"
ref_pred = np.loadtxt(ref_pred_path,dtype=np.float64,delimiter=",")

sam_pred_path = "/ailab/user/tangyuhang/ws/Resultimulate/DifferentEncoderDecoder/simulate_EncoderDecoder/U_Net/4/Simulate/2024-10-15-16-40/sam/img_txt_folder/1800_PredPhawrap2pi.txt"
sam_pred = np.loadtxt(sam_pred_path,dtype=np.float64,delimiter=",")

mcfCore = np.loadtxt('/ailab/user/tangyuhang/ws/Traindata/simulate_data/2pi/15000/5/None/1920-2560/15000_amp_simulate.txt',dtype=np.float32,delimiter=',') 

diff = np.mod((sam_pred-ref_pred-3),2*np.pi)
# diff = (sam_pred-ref_pred)

saveFolder = "/ailab/user/tangyuhang/ws/Resultimulate/DifferentEncoderDecoder/simulate_EncoderDecoder/U_Net/4/Simulate/2024-10-15-16-40"
plt.figure()
plt.imshow(diff)
plt.colorbar()
plt.savefig(f"{saveFolder}/diffCore08.png",dpi=800)

np.savetxt(f"{saveFolder}/diffCore08.txt",diff,fmt="%.10e",delimiter=",")


