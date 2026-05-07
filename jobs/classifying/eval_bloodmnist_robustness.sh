#!/bin/bash

#SBATCH --job-name=eval_bloodmnist_robustness
#SBATCH --output=/scratch/aan266/project/jobs/logs/eval_bloodmnist_robustness_%j.out
#SBATCH --error=/scratch/aan266/project/jobs/logs/eval_bloodmnist_robustness_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch/aan266/project

module load anaconda3

source activate /scratch/aan266/conda_envs/cells-gpu

python -m src.eval.eval_bloodmnist_robustness