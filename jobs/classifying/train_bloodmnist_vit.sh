#!/bin/bash
#SBATCH --job-name=bloodmnist_vit
#SBATCH --output=jobs/logs/bloodmnist_vit_%j.out
#SBATCH --error=jobs/logs/bloodmnist_vit_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

cd /scratch/aan266/project

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cells-gpu

python -m src.train.train_bloodmnist_vit