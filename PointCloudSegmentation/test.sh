#!/bin/bash

#SBATCH --output=output.out
#SBATCH --open-mode=append 
#SBATCH --time=6-0:0:0 
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:12G 

python train.py task=panoptic data=panoptic/npm3d-sparseconv_grid_012_R_8_area4 models=panoptic/area4_ablation_3heads_6 model_name=PointGroup-PAPER training=ablation_area4_head3_6 job_name=ablation_area4_head3_6