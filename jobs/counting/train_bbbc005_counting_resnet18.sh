#!/bin/bash

#SBATCH --job-name=bbbc005_resnet_count
#SBATCH --output=/scratch/aan266/project/jobs/counting/logs/bbbc005_resnet_count_%j.out
#SBATCH --error=/scratch/aan266/project/jobs/counting/logs/bbbc005_resnet_count_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch/aan266/project

module load anaconda3

source activate /scratch/aan266/conda_envs/cells-gpu

python -m src_counting.train.train_bbbc005_counting_resnet18