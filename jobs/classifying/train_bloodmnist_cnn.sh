#!/bin/bash
#SBATCH --job-name=bloodmnist_cnn
#SBATCH --output=jobs/logs/bloodmnist_cnn_%j.out
#SBATCH --error=jobs/logs/bloodmnist_cnn_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

cd /scratch/aan266/project

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cells

python -m src.train.train_bloodmnist