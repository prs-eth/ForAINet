#!/bin/bash

#SBATCH --output=eval_stats.out
#SBATCH --job-name=eval_stats
#SBATCH --open-mode=truncate 
#SBATCH --time=1-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

WANDB_START_METHOD=fork python evaluation_stats.py