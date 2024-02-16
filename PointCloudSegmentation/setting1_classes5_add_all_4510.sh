#!/bin/bash

#SBATCH --output=setting1_classes5_add_all_4510.out
#SBATCH --job-name=setting1_classes5_add_all_4510
#SBATCH --open-mode=truncate 
#SBATCH --time=10-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20G 
#SBATCH --cpus-per-task=4

python train.py task=panoptic data=panoptic/treeins_set1_add_all_4510 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_addallFea_4510 job_name=setting1_classes5_add_all_4510