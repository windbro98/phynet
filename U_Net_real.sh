#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -p optimal
#SBATCH -A optimal
python /ailab/user/tangyuhang/ws/phynet-pro/train_EncoderDecoder_UNET_real.py
