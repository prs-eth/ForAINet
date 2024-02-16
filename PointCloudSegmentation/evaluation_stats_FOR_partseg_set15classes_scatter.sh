#!/bin/bash

#SBATCH --output=evaluation_stats_FOR_partseg_set15classes_scatter.out
#SBATCH --job-name=evaluation_stats_FOR_partseg_set15classes_scatter
#SBATCH --open-mode=truncate 
#SBATCH --time=2-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

WANDB_START_METHOD=fork python evaluation_stats_FOR_partseg_set15classes_scatter.py