#!/bin/bash

#SBATCH --output=setting1_classes5.out
#SBATCH --job-name=setting1_classes5
#SBATCH --open-mode=truncate 
#SBATCH --time=10-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20G 
#SBATCH --cpus-per-task=8

python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1 job_name=setting1_classes5