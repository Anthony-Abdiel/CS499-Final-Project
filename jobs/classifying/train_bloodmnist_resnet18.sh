#!/bin/bash
#SBATCH --job-name=bloodmnist_resnet18
#SBATCH --output=jobs/logs/bloodmnist_resnet18_%j.out
#SBATCH --error=jobs/logs/bloodmnist_resnet18_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

cd /scratch/aan266/project

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cells-gpu

python -m src.train.train_bloodmnist_resnet18