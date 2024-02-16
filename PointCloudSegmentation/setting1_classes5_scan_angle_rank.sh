#!/bin/bash

#SBATCH --output=setting1_classes5_scan_angle_rank.out
#SBATCH --job-name=setting1_classes5_scan_angle_rank
#SBATCH --open-mode=truncate 
#SBATCH --time=10-0:0:0 
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20G 
#SBATCH --cpus-per-task=4

python train.py task=panoptic data=panoptic/treeins_set1_add_scan_angle_rank models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_scan_angle_rank job_name=setting1_classes5_scan_angle_rank