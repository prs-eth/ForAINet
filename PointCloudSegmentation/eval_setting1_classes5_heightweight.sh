#!/bin/bash

#SBATCH --output=eval_setting1_classes5_heightweight.out
#SBATCH --job-name=eval_setting1_classes5_heightweight
#SBATCH --open-mode=truncate 
#SBATCH --time=1-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:12G 
#SBATCH --cpus-per-task=4

WANDB_START_METHOD=fork python eval_setting1_classes5_heightweight.py