#!/bin/bash

#SBATCH --output=scorenettiny_15_mlp_long.out
#SBATCH --job-name=scorenettiny_15_mlp
#SBATCH --open-mode=truncate 
#SBATCH --time=10-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20G 
#SBATCH --cpus-per-task=4

python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads_mlpScore model_name=PointGroup-PAPER training=treeins_set1_mlpScore job_name=scorenettiny_15_mlp