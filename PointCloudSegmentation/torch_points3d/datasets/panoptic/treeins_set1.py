import numpy as np
import torch
import random

from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.datasets.segmentation.treeins_set1 import TreeinsSphere, TreeinsCylinder, INV_OBJECT_LABEL
import torch_points3d.core.data_transform as cT
# from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_points3d.metrics.panoptic_tracker_npm3d import MyPanopticTracker
from torch_points3d.metrics.panoptic_tracker_pointgroup_treeins_partseg import PanopticTracker
from torch_points3d.datasets.panoptic.utils import set_extra_labels
from plyfile import PlyData, PlyElement
import os
from scipy import stats
from torch_points3d.models.panoptic.ply import read_ply, write_ply

# @Treeins: almost all code copied from torch_points3d/datasets/panoptic/npm3d.py, for changes see @Treeins

# @Treeins
# -1 means unlabelled semantic classes
CLASSES_INV = {
    0: "low_vegetation",
    1: "ground",
    2: "stem_points",
    3: "live_branches",
    4: "branches",
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'low_vegetation' .-> .yellow
        [95, 156, 196],  # 'ground' .-> . blue
        [179, 116, 81],  # 'stem_points'  ->  brown
        [241, 149, 131],  # 'live_branches'  ->  salmon
        [81, 163, 148],  # 'branches'  ->  bluegreen
        [0, 0, 0],  # unlabelled .->. black
    ]
)

VALID_CLASS_IDS = [0, 1, 2, 3, 4]
SemIDforInstance = np.array([2,3,4])  # only class tree/1 is is a semantic class with instances, namely tree instances. In contrast, semantic class non-tree is a stuff class.


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
    max_instance = np.max(np.asarray(label)).astype(np.int32) + 1
    rd_colors = np.random.randint(255, size=(max_instance, 3), dtype=np.uint8)
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


# @Treeins: added parameter output_file_name because we now save several final evaluation files because we have several test files
def final_eval(pre_sem, pre_ins_offset, pos, gt_sem, gt_ins, output_file_name):
    NUM_CLASSES = 3  #@Treeins: classes unclassified, non-tree and tree
    NUM_CLASSES_sem = 6
    NUM_CLASSES_count = 5  # @Treeins: 2 classes without unclassified
    # class index for instance segmenatation
    ins_classcount = [2]  # @Treeins
    # class index for stuff segmentation
    stuff_classcount = [1]  # @Treeins
    # class index for semantic segmenatation
    sem_classcount = [1, 2, 3, 4, 5]  # @Treeins
    sem_classcount_remove_ground = [1, 3, 4, 5]
    #class index for semantic classes with gt points
    sem_classcount_have = []
    stuff_classes = [1,2]
    thing_classes = [3,4,5]

    # log directory
    # file_path = '/scratch2/torch-points3d/outputs/2021-10-20/06-19-43/eval/2021-10-26_14-27-55/'
    # predicted semantic segmentation file path
    # pred_class_label_filename = file_path+'Semantic_results_forEval.ply'
    # predicted instance segmentation file path
    # pred_ins_label_filename = file_path+'Instance_Offset_results_forEval.ply'

    # Initialize...
    LOG_FOUT = open(output_file_name + '.txt', 'a')  # @Treeins: save evaluation file with name output_file_name

    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)

    # acc and macc
    true_positive_classes = np.zeros(NUM_CLASSES_sem)
    positive_classes = np.zeros(NUM_CLASSES_sem)
    gt_classes = np.zeros(NUM_CLASSES_sem)

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

    # read files
    # data_class = PlyData.read(pred_class_label_filename)
    # data_ins = PlyData.read(pred_ins_label_filename)

    pred_ins_complete = np.asarray(pre_ins_offset).reshape(-1).astype(np.int)
    pred_sem_complete = np.asarray(pre_sem).reshape(-1).astype(np.int) + 1
    gt_ins_complete = np.asarray(gt_ins).reshape(-1).astype(np.int)
    gt_sem_complete = np.asarray(gt_sem).reshape(-1).astype(np.int) + 1

    # idxc = (gt_sem_complete!=0) | (pred_sem_complete!=0)
    idxc = ((gt_sem_complete != 0) & (gt_sem_complete != 1)& (gt_sem_complete != 2)) | ((pred_sem_complete != 0) & (pred_sem_complete != 1)& (pred_sem_complete != 2))
    pred_ins = pred_ins_complete[idxc]
    gt_ins = gt_ins_complete[idxc]
    pred_sem = pred_sem_complete[idxc]
    gt_sem = gt_sem_complete[idxc]
    pos_ins = pos[idxc].detach().cpu().numpy()
    
    # pn semantic mIoU
    for j in range(gt_sem_complete.shape[0]):
        gt_l = int(gt_sem_complete[j])
        pred_l = int(pred_sem_complete[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l == pred_l)

    # semantic results
    iou_list = []
    for i in range(NUM_CLASSES_sem):
        if gt_classes[i] > 0:
            sem_classcount_have.append(i)
            iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
        else:
            iou = 0.0
        iou_list.append(iou)
        
    set1 = set(sem_classcount)
    set2 = set(sem_classcount_have)
    set3 = set1 & set2
    sem_classcount_final = list(set3)

    log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
    #log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
    log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes[sem_classcount_final] / gt_classes[sem_classcount_final])))
    log_string('Semantic Segmentation IoU: {}'.format(iou_list))
    log_string('Semantic Segmentation mIoU: {}'.format(1.*sum(iou_list)/len(sem_classcount_final)))
    log_string('  ')

    ##remove ground points
    set1 = set(sem_classcount_remove_ground)
    set2 = set(sem_classcount_have)
    set3 = set1 & set2
    sem_classcount_final = list(set3)
    
    log_string('Semantic Segmentation oAcc without ground points: {}'.format(sum(true_positive_classes[sem_classcount_final])/float(sum(positive_classes[sem_classcount_final]))))
    #log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
    log_string('Semantic Segmentation mAcc without ground points: {}'.format(np.mean(np.array(true_positive_classes)[sem_classcount_final] / np.array(gt_classes)[sem_classcount_final])))
    log_string('Semantic Segmentation IoU without ground points: {}'.format(np.array(iou_list)[sem_classcount_final]))
    log_string('Semantic Segmentation mIoU without ground points: {}'.format(1.*sum(np.array(iou_list)[sem_classcount_final])/len(sem_classcount_final)))
    log_string('  ')

    #binary classification results (non_tree and tree)
    #mapping labels
    predicted_labels_copy = pred_sem_complete
    for i in stuff_classes:
        pred_sem_complete[predicted_labels_copy==i] = 1
    for i in thing_classes:
        pred_sem_complete[predicted_labels_copy==i] = 2
        
    gt_labels_copy = gt_sem_complete
    for i in stuff_classes:
        gt_sem_complete[gt_labels_copy==i] = 1
    for i in thing_classes:
        gt_sem_complete[gt_labels_copy==i] = 2
    # pn semantic mIoU
    true_positive_classes = np.zeros(NUM_CLASSES)
    positive_classes = np.zeros(NUM_CLASSES)
    gt_classes = np.zeros(NUM_CLASSES)
    for j in range(gt_sem_complete.shape[0]):
        gt_l = int(gt_sem_complete[j])
        pred_l = int(pred_sem_complete[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l == pred_l)

    # semantic results
    iou_list_bi = []
    sem_classcount_have_bi = []
    for i in range(NUM_CLASSES):
        if gt_classes[i] > 0:
            sem_classcount_have_bi.append(i)
            iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
        else:
            iou = 0.0
        iou_list_bi.append(iou)
        
    sem_classcount_bi = [1, 2] 
    set1 = set(sem_classcount_bi)
    set2 = set(sem_classcount_have_bi)
    set3 = set1 & set2
    sem_classcount_final_bi = list(set3)
    
    set1 = set(stuff_classcount)
    set2 = set(sem_classcount_have_bi)
    set3 = set1 & set2
    stuff_classcount_final = list(set3)

    log_string('Binary Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
    #log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
    log_string('Binary Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes[sem_classcount_final_bi] / gt_classes[sem_classcount_final_bi])))
    log_string('Binary Semantic Segmentation IoU: {}'.format(iou_list_bi))
    log_string('Binary Semantic Segmentation mIoU: {}'.format(1.*sum(iou_list_bi)/len(sem_classcount_final_bi)))
    log_string('  ')
        
    #mapping labels
    predicted_labels_copy = pred_sem
    for i in stuff_classes:
        pred_sem[predicted_labels_copy==i] = 1
    for i in thing_classes:
        pred_sem[predicted_labels_copy==i] = 2
        
    gt_labels_copy = gt_sem
    for i in stuff_classes:
        gt_sem[gt_labels_copy==i] = 1
    for i in thing_classes:
        gt_sem[gt_labels_copy==i] = 2

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
        if g == -1:
            continue
        tmp = (gt_ins == g)
        sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
        pts_in_gt[sem_seg_i] += [tmp]

    # instance mucov & mwcov
    for i_sem in range(NUM_CLASSES):
        sum_cov = 0
        mean_cov = 0
        mean_weighted_cov = 0
        num_gt_point = 0
        if not pts_in_gt[i_sem] or not pts_in_pred[i_sem]:
            all_mean_cov[i_sem].append(0)
            all_mean_weighted_cov[i_sem].append(0)
            continue
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
    if not os.path.exists("viz_for_tp_pre"):
        os.mkdir("viz_for_tp_pre")
    if not os.path.exists("viz_for_fp_pre"):
        os.mkdir("viz_for_fp_pre")
    new_dir_name = "plot" + output_file_name[-6:-4]
    # Construct the full path to the new directory
    viz_for_fp_pre_plot = os.path.join("viz_for_fp_pre", new_dir_name)
    if not os.path.exists(viz_for_fp_pre_plot):
        os.mkdir(viz_for_fp_pre_plot)
    viz_for_tp_pre_plot = os.path.join("viz_for_tp_pre", new_dir_name)
    if not os.path.exists(viz_for_tp_pre_plot):
        os.mkdir(viz_for_tp_pre_plot)
    
    for i_sem in range(NUM_CLASSES):
        if not pts_in_pred[i_sem]:
            continue
        IoU_Tp_per = 0
        IoU_Mc_per = 0
        tp = [0.] * len(pts_in_pred[i_sem])
        fp = [0.] * len(pts_in_pred[i_sem])
        if pts_in_gt[i_sem]:
            total_gt_ins[i_sem] += len(pts_in_gt[i_sem])
        for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
            ovmax = -1.
            if not pts_in_gt[i_sem]:
                fp[ip] = 1
                val_name = os.path.join(viz_for_fp_pre_plot, "sem"+str(i_sem)+"_fp"+str(ip))
                write_ply(val_name,
                    [pos_ins[ins_pred], pred_sem[ins_pred].astype('int32')],
                    ['x', 'y', 'z', 'pre_sem_label'])
                continue
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                union = (ins_pred | ins_gt)
                intersect = (ins_pred & ins_gt)
                iou = float(np.sum(intersect)) / np.sum(union)

                if iou > ovmax:
                    ovmax = iou

            if ovmax > 0:
                IoU_Mc_per += ovmax
            if ovmax >= at:
                tp[ip] = 1  # true
                IoU_Tp_per += ovmax
                #output all tp instances 
                val_name = os.path.join(viz_for_tp_pre_plot, "sem"+str(i_sem)+"_tp"+str(ip))
                write_ply(val_name,
                    [pos_ins[ins_pred], pred_sem[ins_pred].astype('int32')],
                    ['x', 'y', 'z', 'pre_sem_label'])
            else:
                fp[ip] = 1  # false positive
                #output all fp instances
                val_name = os.path.join(viz_for_fp_pre_plot, "sem"+str(i_sem)+"_fp"+str(ip))
                write_ply(val_name,
                    [pos_ins[ins_pred], pred_sem[ins_pred].astype('int32')],
                    ['x', 'y', 'z', 'pre_sem_label'])

        tpsins[i_sem] += tp
        fpsins[i_sem] += fp
        IoU_Tp[i_sem] += IoU_Tp_per
        IoU_Mc[i_sem] += IoU_Mc_per

    MUCov = np.zeros(NUM_CLASSES)
    MWCov = np.zeros(NUM_CLASSES)
    for i_sem in range(NUM_CLASSES):
        MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
        MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])
    
    precision = np.zeros(NUM_CLASSES)
    recall = np.zeros(NUM_CLASSES)
    RQ = np.zeros(NUM_CLASSES)
    SQ = np.zeros(NUM_CLASSES)
    PQ = np.zeros(NUM_CLASSES)
    PQStar = np.zeros(NUM_CLASSES)
    set1 = set(ins_classcount)
    set2 = set(sem_classcount_have)
    set3 = set1 & set2
    ins_classcount_final = list(set3)

    ################################################################
    ######  recall, precision, RQ, SQ, PQ, PQ_star for things ######
    ################################################################
    for i_sem in ins_classcount:
        ###### metrics for offset ######
        if not tpsins[i_sem] or not fpsins[i_sem]:
            continue
        tp = np.asarray(tpsins[i_sem]).astype(np.float)
        fp = np.asarray(fpsins[i_sem]).astype(np.float)
        tp = np.sum(tp)
        fp = np.sum(fp)
        # recall and precision
        if (total_gt_ins[i_sem])==0:
            rec = 0
        else:
            rec = tp / total_gt_ins[i_sem]
        if (tp + fp) == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)
        precision[i_sem] = prec
        recall[i_sem] = rec
        # RQ, SQ, PQ and PQ_star
        if (prec + rec) == 0:
            RQ[i_sem] = 0
        else:
            RQ[i_sem] = 2 * prec * rec / (prec + rec)
        if tp == 0:
            SQ[i_sem] = 0
        else:
            SQ[i_sem] = IoU_Tp[i_sem] / tp   #####
        PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
        # PQStar[i_sem] = IoU_Mc[i_sem]/total_gt_ins[i_sem]
        PQStar[i_sem] = PQ[i_sem]

    ############################################
    ######  RQ, SQ, PQ, PQ_star for stuff ######
    ############################################
    for i_sem in stuff_classcount:
        if iou_list_bi[i_sem] >= 0.5:
            RQ[i_sem] = 1
            SQ[i_sem] = iou_list_bi[i_sem]
        else:
            RQ[i_sem] = 0
            SQ[i_sem] = 0
        PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
        PQStar[i_sem] = iou_list_bi[i_sem]

    if np.mean(precision[ins_classcount_final])+np.mean(recall[ins_classcount_final])==0:
        F1_score = 0.0
    else:
        F1_score = (2*np.mean(precision[ins_classcount_final])*np.mean(recall[ins_classcount_final]))/(np.mean(precision[ins_classcount_final])+np.mean(recall[ins_classcount_final]))
   
    # instance results
    log_string('Instance Segmentation for Offset:')
    log_string('Instance Segmentation MUCov: {}'.format(MUCov[ins_classcount]))
    log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov[ins_classcount_final])))
    log_string('Instance Segmentation MWCov: {}'.format(MWCov[ins_classcount]))
    log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov[ins_classcount_final])))
    log_string('Instance Segmentation Precision: {}'.format(precision[ins_classcount]))
    log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision[ins_classcount_final])))
    log_string('Instance Segmentation Recall: {}'.format(recall[ins_classcount]))
    log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall[ins_classcount_final])))
    log_string('Instance Segmentation F1 score: {}'.format(F1_score))
    log_string('Instance Segmentation RQ: {}'.format(RQ[sem_classcount_bi]))
    log_string('Instance Segmentation meanRQ: {}'.format(np.mean(RQ[sem_classcount_final_bi])))
    log_string('Instance Segmentation SQ: {}'.format(SQ[sem_classcount_bi]))
    log_string('Instance Segmentation meanSQ: {}'.format(np.mean(SQ[sem_classcount_final_bi])))
    log_string('Instance Segmentation PQ: {}'.format(PQ[sem_classcount_bi]))
    log_string('Instance Segmentation meanPQ: {}'.format(np.mean(PQ[sem_classcount_final_bi])))
    log_string('Instance Segmentation PQ star: {}'.format(PQStar[sem_classcount_bi]))
    log_string('Instance Segmentation mean PQ star: {}'.format(np.mean(PQStar[sem_classcount_final_bi])))
    log_string('Instance Segmentation RQ (things): {}'.format(RQ[ins_classcount]))
    log_string('Instance Segmentation meanRQ (things): {}'.format(np.mean(RQ[ins_classcount_final])))
    log_string('Instance Segmentation SQ (things): {}'.format(SQ[ins_classcount]))
    log_string('Instance Segmentation meanSQ (things): {}'.format(np.mean(SQ[ins_classcount_final])))
    log_string('Instance Segmentation PQ (things): {}'.format(PQ[ins_classcount]))
    log_string('Instance Segmentation meanPQ (things): {}'.format(np.mean(PQ[ins_classcount_final])))
    log_string('Instance Segmentation RQ (stuff): {}'.format(RQ[stuff_classcount]))
    log_string('Instance Segmentation meanRQ (stuff): {}'.format(np.mean(RQ[stuff_classcount_final])))
    log_string('Instance Segmentation SQ (stuff): {}'.format(SQ[stuff_classcount]))
    log_string('Instance Segmentation meanSQ (stuff): {}'.format(np.mean(SQ[stuff_classcount_final])))
    log_string('Instance Segmentation PQ (stuff): {}'.format(PQ[stuff_classcount]))
    log_string('Instance Segmentation meanPQ (stuff): {}'.format(np.mean(PQ[stuff_classcount_final])))

class PanopticTreeinsBase:
    INSTANCE_CLASSES = CLASSES_INV.keys()
    NUM_MAX_OBJECTS = 200  # @Treeins: increased int because we had more number of instances in data files from the Treeins data set

    STUFFCLASSES = torch.tensor([i for i in VALID_CLASS_IDS if i not in SemIDforInstance])
    THINGCLASSES = torch.tensor([i for i in VALID_CLASS_IDS if i in SemIDforInstance])
    ID2CLASS = {SemforInsid: i for i, SemforInsid in enumerate(list(SemIDforInstance))}

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
        
        # 3D mix
        #un = torch.max(data['instance_labels']).item()
        #data2 = super().__getitem__(idx)
        #data2['instance_labels'] = data2['instance_labels'] + un
        #for key in data.keys:
        #    if key == 'grid_size':
        #        continue
        #    data[key] = torch.cat((data[key], data2[key]), 0)
            
        #data = data if self.transform is None else self.transform(data)

        # Extract instance and box labels
        self._set_extra_labels(data)
        return data

    def _set_extra_labels(self, data):
        # return set_extra_labels(data, self.INSTANCE_CLASSES, self.NUM_MAX_OBJECTS)
        return set_extra_labels(data, self.ID2CLASS, self.NUM_MAX_OBJECTS)

    def _remap_labels(self, semantic_label):
        return semantic_label

    @property
    def stuff_classes(self):
        # return torch.tensor([0,1,5])
        return self._remap_labels(self.STUFFCLASSES)
    
    @property
    def thing_classes(self):
        # return torch.tensor([0,1,5])
        return self._remap_labels(self.THINGCLASSES)


class PanopticTreeinsSphere(PanopticTreeinsBase, TreeinsSphere):
    def process(self):
        super().process()

    def download(self):
        super().download()


class PanopticTreeinsCylinder(PanopticTreeinsBase, TreeinsCylinder):
    def process(self):
        super().process()

    def download(self):
        super().download()


class TreeinsFusedDataset(BaseDataset):
    """ Wrapper around NPM3DSphere that creates train and test datasets.

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
        dataset_cls = PanopticTreeinsCylinder if sampling_format == "cylinder" else PanopticTreeinsSphere

        # @Treeins: case for training/when running train.py
        if len(self.dataset_opt.fold) == 0 or isinstance(self.dataset_opt.fold[0], int):
            self.train_dataset = dataset_cls(
                self._data_path,
                sample_per_epoch=3000,
                radius=self.dataset_opt.radius,
                grid_size=self.dataset_opt.grid_size,
                forest_regions=self.dataset_opt.forest_regions,  # @Treeins
                add_input_features=self.dataset_opt.add_input_features,  # @Treeins
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
                forest_regions=self.dataset_opt.forest_regions,  # @Treeins
                add_input_features=self.dataset_opt.add_input_features,  # @Treeins
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
                forest_regions=self.dataset_opt.forest_regions,  # @Treeins
                add_input_features=self.dataset_opt.add_input_features,  # @Treeins
                test_area=self.dataset_opt.fold,
                split="test",
                pre_collate_transform=self.pre_collate_transform,
                transform=self.test_transform,
                keep_instance=True,
            )
        # @Treeins: case for evaluation/when running eval.py
            #####xbb
            if dataset_opt.class_weight_method:
                self.add_weights(class_weight_method=dataset_opt.class_weight_method)
        else:
            self.test_dataset = dataset_cls(
                self._data_path,
                sample_per_epoch=-1,
                radius=self.dataset_opt.radius,
                grid_size=self.dataset_opt.grid_size,
                forest_regions=self.dataset_opt.forest_regions,  # @Treeins
                add_input_features=self.dataset_opt.add_input_features,  # @Treeins
                test_area=self.dataset_opt.fold,
                split="test",
                pre_collate_transform=self.pre_collate_transform,
                transform=self.test_transform,
                keep_instance=True,
            )

        # if dataset_opt.class_weight_method:
        #    self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @property
    def test_data_spheres(self):
        return self.test_dataset[0]._test_spheres

    @property
    def test_data_num_spheres(self):
        return self.test_dataset[0]._num_spheres

    @property  # type: ignore
    @save_used_properties
    def stuff_classes(self):
        """ Returns a list of classes that are not instances
        """
        # return self.train_dataset.stuff_classes
        if self.train_dataset:
            return self.train_dataset.stuff_classes
        else:
            return self.test_dataset[0].stuff_classes
        
    @property  # type: ignore
    @save_used_properties
    def thing_classes(self):
        """ Returns a list of classes that are not instances
        """
        # return self.train_dataset.thing_classes
        if self.train_dataset:
            return self.train_dataset.thing_classes
        else:
            return self.test_dataset[0].thing_classes

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save npm3d predictions to disk using s3dis color scheme

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
        """ Allows to save npm3d predictions to disk for evaluation

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
        """ Allows to save npm3d instance predictions to disk using random color

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
    def final_eval(pre_sem, pre_ins_offset, pos, gt_sem, gt_ins, output_file_name):  # @Treeins

        final_eval(pre_sem, pre_ins_offset, pos, gt_sem, gt_ins, output_file_name)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        # return MyPanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
