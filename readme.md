# Automated forest inventory: analysis of high-density airborne LiDAR point clouds with 3D deep learning

This repository represents the official code for paper entitled "Automated forest inventory: analysis of high-density airborne LiDAR point clouds with 3D deep learning".

# Set up environment

Please refer to our previous repo:

https://github.com/prs-eth/PanopticSegForLargeScalePointCloud

It includes the detailed steps and issues that might happen but already resolved.

# FOR-Instance dataset

Please replace the old raw files with our new raw files:

For example, data_set1_5classes contains the data for "basic setting" in Table 4 in our paper.

1. dataset for settings "basic setting", "+ binary semantic loss", "+ class weights", "+ height weights", "+ region weights", "+ elastic distortion and subsampling", "+ TreeMix"

You can download it:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10828636.svg)](https://doi.org/10.5281/zenodo.10828636)

2. For other setting to be added here.

# Commands for running point cloud segmentation experiments based on different settings:

```bash
cd /$YOURPATH$/ForAINet/PointCloudSegmentation
```

1. Experiment for "basic setting" in the paper.

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1 job_name=#YOUR_JOB_NAME#
```

2. Experiment for "+ binary semantic loss" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads_BiLoss model_name=PointGroup-PAPER training=treeins_set1_addBiLoss job_name=#YOUR_JOB_NAME#
```

3. Experiment for "+ class weights" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_classweight models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_nw8_classweight job_name=#YOUR_JOB_NAME#
```

4. Experiment for "+ height weights" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_classweight models=panoptic/FORpartseg_3heads_heightweight model_name=PointGroup-PAPER training=treeins_set1_heightweight job_name=#YOUR_JOB_NAME#
```

5. Experiment for "+ region weights" setting in the paper 

```bash
# Command for training

# To be added
```

6. Experiment for "+ intensity" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_add_intensity models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_intensity job_name=#YOUR_JOB_NAME#
```

7. Experiment for "+ return number" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_add_return_num models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_return_num job_name=#YOUR_JOB_NAME#
```

8. Experiment for "+ scan angle rank" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_add_scan_angle_rank models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_scan_angle_rank job_name=#YOUR_JOB_NAME#
```

9.  Experiment for "+ hand-crafted features" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_add_all_20010 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_addallFea_20010 job_name=#YOUR_JOB_NAME#
```

10. Experiment for "+ elastic distortion and subsampling" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_curved_subsam models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_addCurvedSubsample job_name=#YOUR_JOB_NAME#
```

11. Experiment for "+ TreeMix" setting in the paper 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_treemix3d models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=treeins_set1_mixtree job_name=#YOUR_JOB_NAME#
```

12.  Experiments for data with different point density 

```bash
# Command for training
python train.py task=panoptic data=panoptic/treeins_set1_treemix3d_pd#POINT_DENSITY# models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=mixtree_#POINT_DENSITY# job_name=#YOUR_JOB_NAME#

# take point density=10 as an example
python train.py task=panoptic data=panoptic/treeins_set1_treemix3d_pd10 models=panoptic/FORpartseg_3heads model_name=PointGroup-PAPER training=mixtree_10 job_name=#YOUR_JOB_NAME#
```


13. Commands for testing. Remember to change "checkpoint_dir" parameter to your path.

```bash
# Command for test
# remember to change the following 2 parameters in eval.yaml:
# 1. "checkpoint_dir" to your log files path
# 2. "data" is the paths for your test files
python eval.py

# Command for output the final evaluation file
# replace parameter "test_sem_path" by your path
python evaluation_stats_FOR.py
```

# Commands for running tree parameters extraction code:

```bash
cd /$YOURPATH$/ForAINet/tree_metrics
# remember to adjust parameters based on your dataset
python measurement.py
```

# Commands for running code for extracting manually extracted geometric features:

```bash
# Please note that our code is based on the Superpoint Graphs repository, which can be found at https://github.com/loicland/superpoint_graph. We have included our custom partition_FORdata.py file.
cd /$YOURPATH$/ForAINet/superpoint_graph/partition
python partition_FORdata.py
```

# Citing
If you find our work useful, please do not hesitate to cite it:

```
@article{
  xiang2024automated,
  title={Automated forest inventory: analysis of high-density airborne LiDAR point clouds with 3D deep learning},
  author={Binbin Xiang, Maciej Wielgosz, Theodora Kontogianni, Torben Peters, Stefano Puliti, Rasmus Astrup, Konrad Schindler},
  journal={Remote Sensing of Environment},
  volume={305},
  pages={114078},
  year={2024},
  publisher={Elsevier}
}
```