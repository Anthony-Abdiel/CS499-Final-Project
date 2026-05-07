#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=jobs/logs/test_gpu_%j.out
#SBATCH --error=jobs/logs/test_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

cd /scratch/aan266/project

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cells-gpu

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"