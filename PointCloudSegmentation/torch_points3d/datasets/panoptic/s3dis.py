import numpy as np
import torch
import random

from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.datasets.segmentation.s3dis import S3DISSphere, S3DISCylinder, INV_OBJECT_LABEL
import torch_points3d.core.data_transform as cT
#from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_points3d.metrics.panoptic_tracker_s3dis import MyPanopticTracker
from torch_points3d.metrics.panoptic_tracker_pointgroup import PanopticTracker
from torch_points3d.datasets.panoptic.utils import set_extra_labels
from plyfile import PlyData, PlyElement

import os
import os.path as osp
from scipy import stats
from torch_points3d.models.panoptic.ply import read_ply, write_ply
import time
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
from tqdm.auto import tqdm as tq
import glob
import pandas as pd

CLASSES_INV = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'ceiling' .-> .yellow
        [95, 156, 196],  # 'floor' .-> . blue
        [179, 116, 81],  # 'wall'  ->  brown
        [241, 149, 131],  # 'beam'  ->  salmon
        [81, 163, 148],  # 'column'  ->  bluegreen
        [77, 174, 84],  # 'window'  ->  bright green
        [108, 135, 75],  # 'door'   ->  dark green
        [41, 49, 101],  # 'chair'  ->  darkblue
        [79, 79, 76],  # 'table'  ->  dark grey
        [223, 52, 52],  # 'bookcase'  ->  red
        [89, 47, 95],  # 'sofa'  ->  purple
        [81, 109, 114],  # 'board'   ->  grey
        [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
    ]
)

################################### UTILS #######################################

def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, 'vertex')
    PlyData([el], text=True).write(file)
    print('out')
    
def to_eval_ply(pos, pre_label, gt, file):
    assert len(pre_label.shape) == 1
    assert len(gt.shape) == 1
    assert pos.shape[0] == pre_label.shape[0]
    assert pos.shape[0] == gt.shape[0]
    pos = np.asarray(pos)
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "int16"), ("gt", "int16")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["preds"] = np.asarray(pre_label)
    ply_array["gt"] = np.asarray(gt)
    el = PlyElement.describe(ply_array, 'vertex')
    PlyData([el], text=True).write(file)
    
def to_ins_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    max_instance = np.max(np.asarray(label)).astype(np.int32)+1
    rd_colors = np.random.randint(255, size=(max_instance,3), dtype=np.uint8)
    colors = rd_colors[np.asarray(label).astype(int)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, 'vertex')
    PlyData([el], text=True).write(file)

def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""
    OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label
    
def generate_separate_room(pos, pre_sem, pre_ins_embed, pre_ins_offset):
    raw_dir = '/cluster/work/igp_psr/binbin/torch-points3d/data/s3disfused/raw'
    cloud_names = 'Area_5'
    folders = ["Area_{}".format(i) for i in range(1, 7)]
    test_areas = [f for f in folders if cloud_names in f]
    test_files = [
                    (f, room_name, osp.join(raw_dir, f, room_name))
                    for f in test_areas
                    for room_name in os.listdir(osp.join(raw_dir, f))
                    if os.path.isdir(osp.join(raw_dir, f, room_name))
                ]
    
    t0 = time.time()
    
    #log directory
    #file_path_out = '/scratch2/torch-points3d/outputs/2021-10-13/15-22-11/eval/2021-10-26_00-16-21/'
    #predicted semantic segmentation file path
    #pred_class_label_filenames = file_path_out+'Semantic_results_forEval.ply'
    #predicted instance segmentation file path
    #pred_ins_label_filenames = file_path_out+'Instance_Embed_results_forEval.ply'
    #pred_ins_label_filenames_offset = file_path_out+'Instance_Offset_results_forEval.ply'
    #data_class = PlyData.read(pred_class_label_filenames)
    #data_ins = PlyData.read(pred_ins_label_filenames)
    #data_ins_offset = PlyData.read(pred_ins_label_filenames_offset)
    pred_ins_complete = np.asarray(pre_ins_embed).reshape(-1).astype(np.int)
    pred_ins_complete_offset = np.asarray(pre_ins_offset).reshape(-1).astype(np.int)
    pred_sem_complete = np.asarray(pre_sem).reshape(-1).astype(np.int)
    room_file_path = 'prediction_perRoom_embed' #join(file_path_out +'prediction_perRoom_embed')
    room_file_path2 = 'prediction_perRoom_offset' #join(file_path_out +'prediction_perRoom_offset')
    if not exists(room_file_path):
        makedirs(room_file_path)
    if not exists(room_file_path2):
        makedirs(room_file_path2)
    instance_count = 1
    point_count = 0
    print(test_files)
    for (area, room_name, file_path) in tq(test_files):
    #for cloud_name in cloud_names:
        
        #room_type = room_name.split("_")[0]
        # Initiate containers
        room_points = np.empty((0, 3), dtype=np.float32)
        room_colors = np.empty((0, 3), dtype=np.uint8)
        room_classes = np.empty((0, 1), dtype=np.int32)
        room_instances = np.empty((0, 1), dtype=np.int32)
        room_pre_ins = np.empty((0, 1), dtype=np.int32)
        room_pre_ins_offset = np.empty((0, 1), dtype=np.int32)
        room_pre_classes = np.empty((0, 1), dtype=np.int32)
        objects = glob.glob(osp.join(file_path, "Annotations/*.txt"))
        for single_object in objects:
    
            object_name = os.path.splitext(os.path.basename(single_object))[0]
            object_class = object_name.split("_")[0]
            object_label = object_name_to_label(object_class)
            object_data = pd.read_csv(single_object, sep=" ", header=None).values
    
            # Stack all data
            room_points = np.vstack((room_points, object_data[:, 0:3].astype(np.float32)))
            room_colors = np.vstack((room_colors, object_data[:, 3:6].astype(np.uint8)))
            object_classes = np.full((object_data.shape[0], 1), object_label, dtype=np.int32)
            room_classes = np.vstack((room_classes, object_classes))
            object_instances = np.full((object_data.shape[0], 1), instance_count, dtype=np.int32)
            room_instances = np.vstack((room_instances, object_instances))
            point_num_cur = np.shape(object_data)[0]
            pred_ins_cur = pred_ins_complete[point_count:point_count+point_num_cur].astype(np.uint8).reshape(-1,1)
            room_pre_ins = np.vstack((room_pre_ins, pred_ins_cur))
            pred_ins_cur_offset = pred_ins_complete_offset[point_count:point_count+point_num_cur].astype(np.uint8).reshape(-1,1)
            room_pre_ins_offset = np.vstack((room_pre_ins_offset, pred_ins_cur_offset))
            pred_sem_cur =  pred_sem_complete[point_count:point_count+point_num_cur].astype(np.uint8).reshape(-1,1)
            room_pre_classes = np.vstack((room_pre_classes, pred_sem_cur))
            point_count = point_count + point_num_cur
            instance_count = instance_count + 1
        
        room_file = 'prediction_perRoom_embed/'+ cloud_names+ '_' + room_name + '.ply'
        print(room_file)
        # Save as ply
        write_ply(room_file,
                    (room_points, room_colors, room_classes, room_instances, room_pre_classes, room_pre_ins),
                    ['x', 'y', 'z', 'red', 'green', 'blue', 'gt_class', 'gt_ins', 'pre_sem', 'pre_ins'])
        room_file = 'prediction_perRoom_offset/'+ cloud_names+ '_' + room_name + '.ply'
        print(room_file)
        # Save as ply
        write_ply(room_file,
                    (room_points, room_colors, room_classes, room_instances, room_pre_classes, room_pre_ins_offset),
                    ['x', 'y', 'z', 'red', 'green', 'blue', 'gt_class', 'gt_ins', 'pre_sem', 'pre_ins'])
        print(point_count)
        print(pred_ins_complete.shape)
    
    print('Done in {:.1f}s'.format(time.time() - t0))

def final_eval():
    NUM_CLASSES = 13

    #pred_data_label_filenames = []
    #file_name = '{}/output_filelist.txt'.format(LOG_DIR)
    #pred_data_label_filenames += [line.rstrip() for line in open(file_name)]

    #gt_label_filenames = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_filenames]
    #file_path = '/scratch2/torch-points3d/outputs/2021-10-13/15-22-11/eval/2021-10-26_00-16-21/'
    room_file_path = 'prediction_perRoom_offset'
    room_filesname = listdir(room_file_path) #[join(room_file_path, room) for room in listdir(room_file_path)]
    room_file_path_embed = 'prediction_perRoom_embed'
    #room_filesname_embed = [join(room_file_path_embed, room) for room in listdir(room_file_path_embed)]

    num_room = len(room_filesname)

    # Initialize...
    # acc and macc
    total_true = 0
    total_seen = 0
    true_positive_classes = np.zeros(NUM_CLASSES)
    positive_classes = np.zeros(NUM_CLASSES)
    gt_classes = np.zeros(NUM_CLASSES)
    # mIoU
    ious = np.zeros(NUM_CLASSES)
    totalnums = np.zeros(NUM_CLASSES)
    # precision & recall
    total_gt_ins = np.zeros(NUM_CLASSES)
    at = 0.5
    tpsins = [[] for itmp in range(NUM_CLASSES)]
    fpsins = [[] for itmp in range(NUM_CLASSES)]
    IoU_Tp = np.zeros(NUM_CLASSES)
    IoU_Mc = np.zeros(NUM_CLASSES)
    # mucov and mwcov
    all_mean_cov = [[] for itmp in range(NUM_CLASSES)]
    all_mean_weighted_cov = [[] for itmp in range(NUM_CLASSES)]
    
    #for embeddings
    # precision & recall
    tpsins_embed = [[] for itmp in range(NUM_CLASSES)]
    fpsins_embed = [[] for itmp in range(NUM_CLASSES)]
    IoU_Tp_embed = np.zeros(NUM_CLASSES)
    IoU_Mc_embed = np.zeros(NUM_CLASSES)
    # mucov and mwcov
    all_mean_cov_embed = [[] for itmp in range(NUM_CLASSES)]
    all_mean_weighted_cov_embed = [[] for itmp in range(NUM_CLASSES)]

    for i, room_name in enumerate(room_filesname):
        room_filesname = join(room_file_path, room_name)
        print(room_filesname)
        room_filesname_embed = join(room_file_path_embed, room_name)
        print(room_filesname_embed)
        data_class_cur = read_ply(room_filesname)
        data_class_cur_embed = read_ply(room_filesname_embed)
        pred_ins = data_class_cur['pre_ins'].reshape(-1).astype(np.int)
        pred_sem = data_class_cur['pre_sem'].reshape(-1).astype(np.int)
        gt_ins = data_class_cur['gt_ins'].reshape(-1).astype(np.int)
        gt_sem = data_class_cur['gt_class'].reshape(-1).astype(np.int)
        pred_ins_embed = data_class_cur_embed['pre_ins'].reshape(-1).astype(np.int)
        
        print(gt_sem.shape)

        # semantic acc
        total_true += np.sum(pred_sem == gt_sem)
        total_seen += pred_sem.shape[0]

        # pn semantic mIoU
        for j in range(gt_sem.shape[0]):
            gt_l = int(gt_sem[j])
            pred_l = int(pred_sem[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

        # instance
        un = np.unique(pred_ins)
        pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = (pred_ins == g)
            sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]

        un = np.unique(gt_ins)
        pts_in_gt = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):
            tmp = (gt_ins == g)
            sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
            pts_in_gt[sem_seg_i] += [tmp]

        # instance mucov & mwcov
        for i_sem in range(NUM_CLASSES):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                ovmax = 0.
                num_ins_gt_point = np.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        ipmax = ip

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                all_mean_cov[i_sem].append(mean_cov)

                mean_weighted_cov /= num_gt_point
                all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

        # instance precision & recall
        for i_sem in range(NUM_CLASSES):
            IoU_Tp_per=0
            IoU_Mc_per=0
            tp = [0.] * len(pts_in_pred[i_sem])
            fp = [0.] * len(pts_in_pred[i_sem])
            gtflag = np.zeros(len(pts_in_gt[i_sem]))
            total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                ovmax = -1.

                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        igmax = ig

                if ovmax > 0:
                    IoU_Mc_per += ovmax
                if ovmax >= at:
                    tp[ip] = 1  # true
                    IoU_Tp_per += ovmax
                else:
                    fp[ip] = 1  # false positive

            tpsins[i_sem] += tp
            fpsins[i_sem] += fp
            IoU_Tp[i_sem] += IoU_Tp_per
            IoU_Mc[i_sem] += IoU_Mc_per
        
        # instance for embeddings
        un = np.unique(pred_ins_embed)
        pts_in_pred_embed = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = (pred_ins_embed == g)
            sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
            pts_in_pred_embed[sem_seg_i] += [tmp]

        # instance mucov & mwcov
        for i_sem in range(NUM_CLASSES):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                ovmax = 0.
                num_ins_gt_point = np.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ip, ins_pred in enumerate(pts_in_pred_embed[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        ipmax = ip

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                all_mean_cov_embed[i_sem].append(mean_cov)

                mean_weighted_cov /= num_gt_point
                all_mean_weighted_cov_embed[i_sem].append(mean_weighted_cov)

        # instance precision & recall
        for i_sem in range(NUM_CLASSES):
            IoU_Tp_per=0
            IoU_Mc_per=0
            tp = [0.] * len(pts_in_pred_embed[i_sem])
            fp = [0.] * len(pts_in_pred_embed[i_sem])
            gtflag = np.zeros(len(pts_in_gt[i_sem]))
            #total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

            for ip, ins_pred in enumerate(pts_in_pred_embed[i_sem]):
                ovmax = -1.

                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        igmax = ig

                if ovmax > 0:
                    IoU_Mc_per += ovmax
                if ovmax >= at:
                    tp[ip] = 1  # true
                    IoU_Tp_per += ovmax
                else:
                    fp[ip] = 1  # false positive

            tpsins_embed[i_sem] += tp
            fpsins_embed[i_sem] += fp
            IoU_Tp_embed[i_sem] += IoU_Tp_per
            IoU_Mc_embed[i_sem] += IoU_Mc_per

    MUCov = np.zeros(NUM_CLASSES)
    MWCov = np.zeros(NUM_CLASSES)
    MUCov_embed = np.zeros(NUM_CLASSES)
    MWCov_embed = np.zeros(NUM_CLASSES)
    for i_sem in range(NUM_CLASSES):
        MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
        MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])
        MUCov_embed[i_sem] = np.mean(all_mean_cov_embed[i_sem])
        MWCov_embed[i_sem] = np.mean(all_mean_weighted_cov_embed[i_sem])

    precision = np.zeros(NUM_CLASSES)
    recall = np.zeros(NUM_CLASSES)
    precision_embed = np.zeros(NUM_CLASSES)
    recall_embed = np.zeros(NUM_CLASSES)
    RQ = np.zeros(NUM_CLASSES)
    SQ = np.zeros(NUM_CLASSES)
    PQ = np.zeros(NUM_CLASSES)
    #PQStar = np.zeros(NUM_CLASSES)
    RQ_embed = np.zeros(NUM_CLASSES)
    SQ_embed = np.zeros(NUM_CLASSES)
    PQ_embed = np.zeros(NUM_CLASSES)
    #PQStar_embed = np.zeros(NUM_CLASSES)
    for i_sem in range(NUM_CLASSES):
        tp = np.asarray(tpsins[i_sem]).astype(np.float)
        fp = np.asarray(fpsins[i_sem]).astype(np.float)
        tp = np.sum(tp)
        fp = np.sum(fp)
        rec = tp / total_gt_ins[i_sem]
        prec = tp / (tp + fp)

        precision[i_sem] = prec
        recall[i_sem] = rec
        RQ[i_sem] = 2*prec*rec/(prec+rec)
        if (prec+rec)==0:
            RQ[i_sem] = 0
        SQ[i_sem] = IoU_Tp[i_sem]/tp
        if tp==0:
            SQ[i_sem] = 0
        PQ[i_sem] = SQ[i_sem]*RQ[i_sem]
        #PQStar[i_sem] = IoU_Mc[i_sem]/total_gt_ins[i_sem]
        
        tp = np.asarray(tpsins_embed[i_sem]).astype(np.float)
        fp = np.asarray(fpsins_embed[i_sem]).astype(np.float)
        tp = np.sum(tp)
        fp = np.sum(fp)
        rec = tp / total_gt_ins[i_sem]
        prec = tp / (tp + fp)

        precision_embed[i_sem] = prec
        recall_embed[i_sem] = rec
        RQ_embed[i_sem] = 2*prec*rec/(prec+rec)
        if (prec+rec)==0:
            RQ_embed[i_sem] = 0
        SQ_embed[i_sem] = IoU_Tp_embed[i_sem]/tp
        if tp==0:
            SQ_embed[i_sem] = 0
        PQ_embed[i_sem] = SQ_embed[i_sem]*RQ_embed[i_sem]
        #PQStar_embed[i_sem] = IoU_Mc_embed[i_sem]/total_gt_ins[i_sem]

    LOG_FOUT = open('evaluation.txt', 'a')

    F1_score = (2*np.mean(precision)*np.mean(recall))/(np.mean(precision)+np.mean(recall))
    F1_score_embed = (2*np.mean(precision_embed)*np.mean(recall_embed))/(np.mean(precision_embed)+np.mean(recall_embed))
    
    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)


    # instance results
    log_string('Instance Segmentation for Offset:')
    log_string('Instance Segmentation MUCov: {}'.format(MUCov.tolist()))
    log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov)))
    log_string('Instance Segmentation MWCov: {}'.format(MWCov.tolist()))
    log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov)))
    log_string('Instance Segmentation Precision: {}'.format(precision.tolist()))
    log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision)))
    log_string('Instance Segmentation Recall: {}'.format(recall.tolist()))
    log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall)))
    log_string('Instance Segmentation F1 score: {}'.format(F1_score))
    log_string('Instance Segmentation RQ: {}'.format(RQ))
    log_string('Instance Segmentation meanRQ: {}'.format(np.mean(RQ)))
    log_string('Instance Segmentation SQ: {}'.format(SQ))
    log_string('Instance Segmentation meanSQ: {}'.format(np.mean(SQ)))
    log_string('Instance Segmentation PQ: {}'.format(PQ))
    log_string('Instance Segmentation meanPQ: {}'.format(np.mean(PQ)))
    #log_string('Instance Segmentation PQ star: {}'.format(PQStar))
    #log_string('Instance Segmentation mean PQ star: {}'.format(np.mean(PQStar)))
    
    log_string('Instance Segmentation for Embeddings:')
    log_string('Instance Segmentation MUCov: {}'.format(MUCov_embed.tolist()))
    log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov_embed)))
    log_string('Instance Segmentation MWCov: {}'.format(MWCov_embed.tolist()))
    log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov_embed)))
    log_string('Instance Segmentation Precision: {}'.format(precision_embed.tolist()))
    log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision_embed)))
    log_string('Instance Segmentation Recall: {}'.format(recall_embed.tolist()))
    log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall_embed)))
    log_string('Instance Segmentation F1 score: {}'.format(F1_score_embed))
    log_string('Instance Segmentation RQ: {}'.format(RQ_embed))
    log_string('Instance Segmentation meanRQ: {}'.format(np.mean(RQ_embed)))
    log_string('Instance Segmentation SQ: {}'.format(SQ_embed))
    log_string('Instance Segmentation meanSQ: {}'.format(np.mean(SQ_embed)))
    log_string('Instance Segmentation PQ: {}'.format(PQ_embed))
    log_string('Instance Segmentation meanPQ: {}'.format(np.mean(PQ_embed)))
    #log_string('Instance Segmentation PQ star: {}'.format(PQStar_embed))
    #log_string('Instance Segmentation mean PQ star: {}'.format(np.mean(PQStar_embed)))

    # semantic results
    iou_list = []
    for i in range(NUM_CLASSES):
        iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        # print(iou)
        iou_list.append(iou)

    log_string('Semantic Segmentation:')
    log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes) / float(sum(positive_classes))))
    # log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
    log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes / gt_classes)))
    log_string('Semantic Segmentation IoU: {}'.format(iou_list))
    log_string('Semantic Segmentation mIoU: {}'.format(1. * sum(iou_list) / NUM_CLASSES))

class PanopticS3DISBase:
    INSTANCE_CLASSES = CLASSES_INV.keys()
    NUM_MAX_OBJECTS = 100

    def __getitem__(self, idx):
        """
        Data object contains:
            pos - points
            x - features
        """
        if not isinstance(idx, int):
            raise ValueError("Only integer indices supported")

        # Get raw data and apply transforms
        data = super().__getitem__(idx)

        # Extract instance and box labels
        self._set_extra_labels(data)
        return data

    def _set_extra_labels(self, data):
        return set_extra_labels(data, self.INSTANCE_CLASSES, self.NUM_MAX_OBJECTS)

    @property
    def stuff_classes(self):
        return torch.tensor([])


class PanopticS3DISSphere(PanopticS3DISBase, S3DISSphere):
    def process(self):
        super().process()

    def download(self):
        super().download()


class PanopticS3DISCylinder(PanopticS3DISBase, S3DISCylinder):
    def process(self):
        super().process()

    def download(self):
        super().download()


class S3DISFusedDataset(BaseDataset):
    """ Wrapper around S3DISSphere that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = PanopticS3DISCylinder if sampling_format == "cylinder" else PanopticS3DISSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            radius=self.dataset_opt.radius,
            grid_size=self.dataset_opt.grid_size,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
            keep_instance=True,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            radius=self.dataset_opt.radius,
            grid_size=self.dataset_opt.grid_size,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
            keep_instance=True,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            radius=self.dataset_opt.radius,
            grid_size=self.dataset_opt.grid_size,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
            keep_instance=True,
        )

        #if dataset_opt.class_weight_method:
        #    self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @property
    def test_data_spheres(self):
        return self.test_dataset[0]._test_spheres

    @property  # type: ignore
    @save_used_properties
    def stuff_classes(self):
        """ Returns a list of classes that are not instances
        """
        return self.train_dataset.stuff_classes
        
    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    @staticmethod
    def to_eval_ply(pos, pre_label, gt, file):
        """ Allows to save s3dis predictions to disk for evaluation

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        pre_label : torch.Tensor
            predicted label
        gt : torch.Tensor
            instance GT label
        file : string
            Save location
        """
        to_eval_ply(pos, pre_label, gt, file)
        
    @staticmethod
    def to_ins_ply(pos, label, file):
        """ Allows to save s3dis instance predictions to disk using random color

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted instance label
        file : string
            Save location
        """
        to_ins_ply(pos, label, file)


    @staticmethod
    def generate_separate_room(pos, pre_sem, pre_ins_embed, pre_ins_offset):

        generate_separate_room(pos, pre_sem, pre_ins_embed, pre_ins_offset)
    
    @staticmethod
    def final_eval():

        final_eval()
        
    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        #return MyPanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

