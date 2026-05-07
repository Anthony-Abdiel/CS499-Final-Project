#!/bin/bash

#SBATCH --job-name=eval_bbbc005_blur
#SBATCH --output=/scratch/aan266/project/jobs/counting/logs/eval_bbbc005_blur_%j.out
#SBATCH --error=/scratch/aan266/project/jobs/counting/logs/eval_bbbc005_blur_%j.err
#SBATCH --time=00:45:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch/aan266/project

module load anaconda3

source activate /scratch/aan266/conda_envs/cells-gpu

python -m src_counting.eval.eval_bbbc005_robustness