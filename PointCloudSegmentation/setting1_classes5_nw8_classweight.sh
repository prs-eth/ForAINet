#!/bin/bash

#SBATCH --output=setting1_classes5_nw8_classweight.out
#SBATCH --job-name=setting1_classes5_nw8_classweight
#SBATCH --open-mode=truncate 
#SBATCH --time=10-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:24G 
#SBATCH --cpus-per-task=8

python train.py task=panoptic data=panoptic/treeins_set1_classweight models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_nw8_classweight job_name=setting1_classes5_nw8_classweight