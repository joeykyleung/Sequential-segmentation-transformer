#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=<jkl223>

# Set up your Python environment if using a virtual environment or conda
export PATH=/vol/bitbucket/${USER}/cv_venv/bin/:$PATH

# Uncomment and adjust the following line if using miniconda or anaconda
# source ~/.bashrc

source activate

# Source CUDA setup script
source /vol/cuda/12.5.0/setup.sh

# Print GPU information
/usr/bin/nvidia-smi

# Print system uptime
uptime

export CUDA_LAUNCH_BLOCKING=1
# execute Python script
# python3 '/homes/jkl223/Desktop/Individual Project/ACDCDataset.py'
# python3 '/homes/jkl223/Desktop/Individual Project/tokenize_data.py' 'test'

# python3 '/homes/jkl223/Desktop/Individual Project/train.py' -x 'UNET2D' 2000
# python3 '/homes/jkl223/Desktop/Individual Project/train.py' -x 'UNET3D' 20000
# python3 '/homes/jkl223/Desktop/Individual Project/train.py' -x 'UNETR' 20000
# python3 '/homes/jkl223/Desktop/Individual Project/train.py' 'VQGAN' 5000

python3 '/homes/jkl223/Desktop/Individual Project/train.py' 'SST' 5000 -t
# python3 '/homes/jkl223/Desktop/Individual Project/train.py' 'SST' 500

