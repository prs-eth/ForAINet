#!/bin/bash

#SBATCH --output=eval_density_1000.out
#SBATCH --job-name=eval_density_1000
#SBATCH --open-mode=truncate 
#SBATCH --time=1-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20G 
#SBATCH --cpus-per-task=8

WANDB_START_METHOD=fork python eval_density_1000.py