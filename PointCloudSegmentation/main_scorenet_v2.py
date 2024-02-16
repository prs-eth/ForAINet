import MinkowskiEngine.MinkowskiPooling
import numpy as np
import torch.optim
from network.resnet import *
from network.Network import *
from torch.utils.data import DataLoader
from Dset_set import Dataset
from typing import NamedTuple, Dict, Any, List, Tuple
import wandb
import math
from sklearn.metrics import f1_score,precision_score,recall_score
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from tqdm import tqdm
from torch_geometric.data import Data
from torch_scatter import scatter
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP, FastBatchNorm1d
from network.utils import RandomRotation,RandomScale,RandomShear,RandomTranslation
from typing import List
from torch_points_kernels import instance_iou
import torchnet as tnt
from collections import OrderedDict, defaultdict
from torch_points3d.metrics.box_detection.ap import voc_ap
from torch_scatter import scatter_add
import argparse
import ast
wandb.init(project="Scorenet-ProposalGroup")
option = torch.load('/cluster/work/igp_psr/binbin/OutdoorPanopticSeg_V2/scorenet/option')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='v2')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
#parser.add_argument('--add_augmentation', type=bool, default=False, help='add data augmentation for each proposal during training (default: False)')
#parser.add_argument('--independent', type=bool, default=True, help='not adding proposal grouping step (default: True)')
parser.add_argument('--add_augmentation', type=ast.literal_eval,default=False, dest='add_augmentation', help='add data augmentation for each proposal during training (default: False)')
parser.add_argument('--independent', type=ast.literal_eval, default=True, dest='independent', help='not adding proposal grouping step (default: True)')
parser.add_argument('--clip_min', type=float, default=0, help='min clip value for loss calculation (default: 0)')
parser.add_argument('--clip_max', type=float, default=1, help='max clip value for loss calculation (default: 1)')
args = parser.parse_args()
wandb.config.update(args) # adds all of the arguments as config variables

class _Instance(NamedTuple):
    classname: str
    score: float
    indices: np.array  # type: ignore
    scan_id: int

    def iou(self, other: "_Instance") -> float:
        assert self.scan_id == other.scan_id
        intersection = float(len(np.intersect1d(other.indices, self.indices)))
        return intersection / float(len(other.indices) + len(self.indices) - intersection)

    def find_best_match(self, others: List["_Instance"]) -> Tuple[float, int]:
        ioumax = -np.inf
        best_match = -1
        for i, other in enumerate(others):
            iou = self.iou(other)
            if iou > ioumax:
                ioumax = iou
                best_match = i
        return ioumax, best_match

class InstanceAPMeter:
    def __init__(self):
        self._pred_clusters = defaultdict(list)  # {classname: List[_Instance]}
        self._gt_clusters = defaultdict(lambda: defaultdict(list))  # {classname:{scan_id: List[_Instance]}

    def add(self, pred_clusters: List[_Instance], gt_clusters: List[_Instance]):
        for instance in pred_clusters:
            self._pred_clusters[instance.classname].append(instance)
        for instance in gt_clusters:
            self._gt_clusters[instance.classname][instance.scan_id].append(instance)

    def _eval_cls(self, classname, iou_threshold):
        preds = self._pred_clusters.get(classname, [])
        allgts = self._gt_clusters.get(classname, {})
        visited = {scan_id: len(gt) * [False] for scan_id, gt in allgts.items()}
        ngt = 0
        for gts in allgts.values():
            ngt += len(gts)

        # Start with most confident first
        preds.sort(key=lambda x: x.score, reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for p, pred in enumerate(preds):
            scan_id = pred.scan_id
            gts = allgts.get(scan_id, [])
            if len(gts) == 0:
                fp[p] = 1
                continue

            # Find best macth in ground truth
            ioumax, best_match = pred.find_best_match(gts)

            if ioumax < iou_threshold:
                fp[p] = 1
                continue

            if visited[scan_id][best_match]:
                fp[p] = 1
            else:
                visited[scan_id][best_match] = True
                tp[p] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(ngt)

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        return rec, prec, ap

    def eval(self, iou_threshold, processes=1):
        rec = {}
        prec = {}
        ap = {}
        for classname in self._gt_clusters.keys():
            rec[classname], prec[classname], ap[classname] = self._eval_cls(classname, iou_threshold)

        for i, classname in enumerate(self._gt_clusters.keys()):
            if classname not in self._pred_clusters:
                rec[classname] = 0
                prec[classname] = 0
                ap[classname] = 0

        return rec, prec, ap


class SetResNet(torch.nn.Module):
    def __init__(self): #,independent=False):
        super(SetResNet, self).__init__()
        #old scorenet
        #self.ScorerUnet = Minkowski("unet", input_nc=16, num_layers=4, config=option.scorer_unet)
        #self.ScorerHead = Seq().append(torch.nn.Linear(16, 1)).append(torch.nn.Sigmoid())

        #new scorenet
        self.backbone = MinkUNet14(in_channels=16, out_channels=1, D=3)#ResNet18(in_channels=16, out_channels=1, D=3)
        self.backbone.final = nn.Identity()
        self.backbone.relu = MinkowskiEngine.MinkowskiPooling.MinkowskiGlobalMaxPooling()
        self.classifier = Seq().append(torch.nn.Linear(96, 1)).append(torch.nn.Sigmoid())
        
        self.independent = args.independent
        if not self.independent:
            #self.att = nn.MultiheadAttention(16, 4,batch_first=False,add_bias_kv=False)
            self.att = nn.MultiheadAttention(96, 6,batch_first=False,add_bias_kv=False)

        self.criterion = nn.BCEWithLogitsLoss()
        self.CE = nn.CrossEntropyLoss()

        #self.pose = PositionalEncoder(96)
        #self.inference = torch.nn.Sigmoid()


    def forward(self,pos3d,coords_input,back_feat_input,sem_feat_input,score_gt_input,score_pre,group,all_clusters,inference=False,shuffle_groups=False):

        # Assemble batches
        x = [] # backbone features
        coords = [] # input coords
        batch = [] 
        pos = []
        #data_list = []
        for i, cluster in enumerate(all_clusters):
            
            score_gt = score_gt_input[i]
            back_feat = back_feat_input[cluster]
            #sem_feat = sem_feat_input[cluster]
            
            #data augmentation
            position = pos3d[cluster]
            coords_i = coords_input[cluster]
            
            if args.add_augmentation:
                position -= torch.mean(position)
                if not inference:
                    position,score_gt = RandomRotation()(position,score_gt)
                    position,score_gt = RandomScale(0.85, 1.15)(position,score_gt)
                    position,score_gt = RandomShear()(position,score_gt)
                    position,score_gt = RandomTranslation()(position,score_gt)
                position -= torch.min(position)
                coords_i,back_feat = ME.utils.sparse_quantize(coordinates=position, features=back_feat,quantization_size=0.12)
                #coords_i_1,sem_feat = ME.utils.sparse_quantize(coordinates=position, features=sem_feat,quantization_size=0.06)
                #coords_i, unique_map, inverse_map = ME.utils.sparse_quantize(coordinates=position, quantization_size=0.06, return_index=True, return_inverse=True)
                
            
            #previous
            x.append(back_feat)
            coords.append(coords_i)
            batch.append(i * torch.ones(coords_i.shape[0]))
            #pos.append(position)
            #data_list.append()
        batch_cluster = Data(x=torch.cat(x), coords=torch.cat(coords), batch=torch.cat(batch).int(),).to(device)
        
        #old scorenet
        #score_backbone_out = self.ScorerUnet(batch_cluster)
        #scatter_features = score_backbone_out.x
        
        #new scorenet
        coords_ = torch.cat([batch_cluster.batch.unsqueeze(-1).int(), batch_cluster.coords.int()], -1)
        input = ME.SparseTensor(features=batch_cluster.x, coordinates=coords_) #, device=device)
        scatter_features = self.backbone(input).F
        
        
        uniq = torch.unique(group)
        
        #scatter_features = scatter_features.to("cpu")
        #batch_cluster = batch_cluster.to("cpu")
        if self.independent:
            #cluster_feats = scatter(
            #    scatter_features, batch_cluster.batch.long(), dim=0, reduce="max"
            #) # [num_cluster, 16]

            #cluster_scores = self.ScorerHead(cluster_feats.to(device)).squeeze(-1) # [num_cluster, 1]
            cluster_scores = self.classifier(scatter_features).squeeze(-1) # [num_cluster, 1]
        else:
            #features = scatter_features  #or.F
            #cluster_feats = scatter(
            #    scatter_features, batch_cluster.batch.long(), dim=0, reduce="max"
            #) 
            #all_related_features = torch.zeros_like(cluster_feats)
            all_related_features = torch.zeros_like(scatter_features)
            #batch_now = batch_cluster.batch #score_backbone_out.batch
            #batch_new = []
            #all_related_features = []
            #cluster_idx = 0
            for set in uniq:
                idx = group == set
                #idx_ = idx.nonzero()
                #idx2 = torch.zeros_like(batch_now, dtype=torch.bool)
                #for jj in idx_:
                #    mask_jj = batch_now.cpu() == jj
                #    idx2[mask_jj] = True
                #    #batch_new.append(cluster_idx * torch.ones(idx2[mask_jj].shape[0]))
                ##related_labels.append(labels[idx])
                unrelated_features = scatter_features[idx]
                unrelated_features = unrelated_features.unsqueeze(0)
                #unrelated_features = unrelated_features.to(device)
                related_features,_ = self.att(unrelated_features,unrelated_features,unrelated_features)
                #related_features = related_features.squeeze(0).cpu()
                all_related_features[idx] = related_features
                #all_related_features.append(related_features)
                #all_groups.append(groups[idx2])

            #groups = torch.cat(all_groups,0)
            #labels = torch.cat(related_labels,0)
            #preds = self.classifier(torch.cat(all_related_features,0))
            # [num_cluster, 16]

            #cluster_scores = self.ScorerHead(all_related_features.to(device)).squeeze(-1) # [num_cluster, 1]
            cluster_scores = self.classifier(all_related_features).squeeze(-1) # [num_cluster, 1]
        
        
        
        return cluster_scores

def collate_fn(batch):
    pos3d = [x[0] for x in batch]
    coords = [x[1] for x in batch]
    back_feat = [x[2] for x in batch]
    sem_feat = [x[3] for x in batch]
    score_gt = [x[4] for x in batch]
    score_pre = [x[5] for x in batch]
    group = [x[6] for x in batch]
    all_clusters = [x[7] for x in batch]
    gt_ins = [x[8] for x in batch]
    gt_sem = [x[9] for x in batch]
    batch_idx = []
    num_ins = []
    # coords,feats = ME.utils.sparse_collate(coords,feats)
    # print(coords)
    # exit()
    pointnum_total = 0
    all_clusters_new = []
    for instance in all_clusters[0]:
        all_clusters_new.append(instance)
    for i in range(1,len(all_clusters)):
        pre_coords = coords[i-1]
        pre_points_num = pre_coords.shape[0]
        pointnum_total = pointnum_total + pre_points_num
        for instance in all_clusters[i]:
            instance = pointnum_total + instance
            all_clusters_new.append(instance)

    for i in range(1, len(group)):
        prev_group_max = torch.max(group[i-1])
        group[i] += prev_group_max + 1
    
    for i, j in enumerate(gt_ins):    
        batch_idx.append(i * torch.ones(j.shape[0]).int())
        unique_ins = torch.unique(j)
        valid_ins = unique_ins != 0
        num_ins.append(torch.sum(valid_ins))

    pos3d = torch.cat(pos3d, 0)
    coords = torch.cat(coords, 0)
    back_feat = torch.cat(back_feat,0)
    sem_feat = torch.cat(sem_feat,0)
    score_gt = torch.cat(score_gt,0)
    score_pre = torch.cat(score_pre,0)
    groups = torch.cat(group,0)
    gt_ins = torch.cat(gt_ins,0)
    gt_sem = torch.cat(gt_sem,0)
    batch_idx = torch.cat(batch_idx)
    num_ins = torch.stack(num_ins, dim=0)
    assert len(coords) == len(back_feat)
    assert len(groups) == len(score_gt)
    return pos3d,coords,back_feat,sem_feat,score_gt,score_pre,groups,all_clusters_new, gt_ins, gt_sem, batch_idx, num_ins

import matplotlib.pyplot as plt
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if "backbone" in n:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
    
def cal_score_loss(
    gt_scores: torch.Tensor,
    predicted_clusters: List[torch.Tensor],
    cluster_scores: torch.Tensor,
    batch: torch.Tensor,
    min_iou_threshold=0.25,
    max_iou_threshold=0.75,
):
    """ Loss that promotes higher scores for clusters with higher instance iou,
    see https://arxiv.org/pdf/2004.01658.pdf equation (7)
    """
    assert len(predicted_clusters) == cluster_scores.shape[0]
    ious = gt_scores
    lower_mask = ious < min_iou_threshold
    higher_mask = ious > max_iou_threshold
    middle_mask = torch.logical_and(torch.logical_not(lower_mask), torch.logical_not(higher_mask))
    assert torch.sum(lower_mask + higher_mask + middle_mask) == ious.shape[0]
    shat = torch.zeros_like(ious)
    iou_middle = ious[middle_mask]
    shat[higher_mask] = 1
    shat[middle_mask] = (iou_middle - min_iou_threshold) / (max_iou_threshold - min_iou_threshold)
    return torch.nn.functional.binary_cross_entropy(cluster_scores, shat)

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return pick

def extract_clusters(clusters, cluster_scores, semantic_logits, nms_threshold=0.3, min_cluster_points=10, min_score=0.2):
    #valid_cluster_idx, clusters = outputs.get_instances(min_cluster_points=min_cluster_points)
    """ Returns index of clusters that pass nms test, min size test and score test
    """
    if not clusters:
        return [], []
    
    if cluster_scores==None:
        return None, clusters
    n_prop = len(clusters)
    proposal_masks = torch.zeros(n_prop, semantic_logits.shape[0])
    # for i, cluster in enumerate(self.clusters):
    #     proposal_masks[i, cluster] = 1
    
    proposals_idx = []
    for i, cluster in enumerate(clusters):
        proposal_id = torch.ones(len(cluster))*i
        proposals_idx.append(torch.vstack((proposal_id,cluster)).T)
    proposals_idx = torch.cat(proposals_idx, dim=0)
    
    proposals_idx_filtered = proposals_idx
    #proposals_idx_filtered = proposals_idx[_mask]
    proposal_masks[proposals_idx_filtered[:, 0].long(), proposals_idx_filtered[:, 1].long()] = 1

    intersection = torch.mm(proposal_masks, proposal_masks.t())  # (nProposal, nProposal), float, cuda
    proposals_pointnum = proposal_masks.sum(1)  # (nProposal), float, cuda
    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), cluster_scores.cpu().numpy(), nms_threshold)

    valid_pick_ids = []
    valid_clusters = []
    for i in pick_idxs:
        cl_mask = proposals_idx_filtered[:,0]==i
        cl = proposals_idx_filtered[cl_mask][:,1].long()
        if len(cl) > min_cluster_points and cluster_scores[i] > min_score:
            valid_pick_ids.append(i)
            valid_clusters.append(cl)
    return valid_clusters, valid_pick_ids

def compute_acc(clusters, predicted_labels, instance_labels, y, batch, num_instances, iou_threshold):
    """ Computes the ratio of True positives, False positives and accuracy
    """
    iou_values, gt_ids = instance_iou(clusters, instance_labels, batch.int()).max(1)
    gt_ids += 1
    instance_offsets = torch.cat((torch.tensor([0]), num_instances.cumsum(-1)))
    tp = 0
    fp = 0
    for i, iou in enumerate(iou_values):
        # Too low iou, no match in ground truth
        if iou < iou_threshold:
            fp += 1
            continue

        # Check that semantic is correct
        sample_idx = batch[clusters[i][0]]
        sample_mask = batch == sample_idx
        instance_offset = instance_offsets[sample_idx]
        gt_mask = instance_labels[sample_mask] == (gt_ids[i] - instance_offset)
        gt_classes = y[sample_mask][torch.nonzero(gt_mask, as_tuple=False)]
        gt_classes, counts = torch.unique(gt_classes, return_counts=True)
        gt_class = gt_classes[counts.max(-1)[1]]
        pred_class = torch.mode(predicted_labels[clusters[i]])
        #predicted_labels[clusters[i][0]]
        if gt_class == pred_class[0]:
            tp += 1
        else:
            fp += 1
    acc = tp / len(clusters)
    #num_instances = torch.unique(instance_labels).shape[0]
    tp = tp / torch.sum(num_instances).cpu().item()
    fp = fp / torch.sum(num_instances).cpu().item()
    return tp, fp, acc

def compute_eval(clusters, predicted_labels, instance_labels, y , batch, num_instances, num_classe, iou_threshold):
        
    ins_classcount_have = []
    pts_in_pred = [[] for itmp in range(num_classe)]
    for g in clusters:  # each object in prediction
        tmp = torch.zeros_like(predicted_labels, dtype=torch.bool)
        tmp[g] = True
        sem_seg_i = int(torch.mode(predicted_labels[tmp])[0])
        pts_in_pred[sem_seg_i] += [tmp]
        
    pts_in_gt = [[] for itmp in range(num_classe)]
    unique_in_batch = torch.unique(batch)
    for s in unique_in_batch:
        batch_mask = batch == s
        un = torch.unique(instance_labels[batch_mask])
        for ig, g in enumerate(un):
            if g == -1:
                continue
            tmp = (instance_labels == g) & batch_mask.to(num_instances.device)
            sem_seg_i = int(torch.mode(y[tmp])[0])
            if sem_seg_i == -1:
                continue
            pts_in_gt[sem_seg_i] += [tmp]
            ins_classcount_have.append(sem_seg_i)
        
    all_mean_cov = [[] for itmp in range(num_classe)]
    all_mean_weighted_cov = [[] for itmp in range(num_classe)]
    # instance mucov & mwcov
    for i_sem in range(num_classe):
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
            num_ins_gt_point = torch.sum(ins_gt)
            num_gt_point += num_ins_gt_point
            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                union = (ins_pred | ins_gt)
                intersect = (ins_pred & ins_gt)
                iou = float(torch.sum(intersect)) / torch.sum(union)

                if iou >= ovmax:
                    ovmax = iou
                    ipmax = ip

            sum_cov += ovmax
            mean_weighted_cov += ovmax * num_ins_gt_point

        if len(pts_in_gt[i_sem]) != 0:
            mean_cov = sum_cov / len(pts_in_gt[i_sem])
            all_mean_cov[i_sem].append(mean_cov.item())

            mean_weighted_cov /= num_gt_point
            all_mean_weighted_cov[i_sem].append(mean_weighted_cov.item())

    #print(all_mean_cov)
    total_gt_ins = np.zeros(num_classe)
    at = iou_threshold
    tpsins = [[] for itmp in range(num_classe)]
    fpsins = [[] for itmp in range(num_classe)]
    IoU_Tp = np.zeros(num_classe)
    IoU_Mc = np.zeros(num_classe)
    # instance precision & recall
    for i_sem in range(num_classe):
        if not pts_in_pred[i_sem]:
            continue
        IoU_Tp_per=0
        IoU_Mc_per=0
        tp = [0.] * len(pts_in_pred[i_sem])
        fp = [0.] * len(pts_in_pred[i_sem])
        #gtflag = np.zeros(len(pts_in_gt[i_sem]))
        if pts_in_gt[i_sem]:
            total_gt_ins[i_sem] += len(pts_in_gt[i_sem])
        for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
            ovmax = -1.
            if not pts_in_gt[i_sem]:
                fp[ip] = 1
                continue
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                union = (ins_pred | ins_gt)
                intersect = (ins_pred & ins_gt)
                iou = (float(torch.sum(intersect)) / torch.sum(union)).item()

                if iou > ovmax:
                    ovmax = iou
                    #igmax = ig

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
    
    MUCov = torch.zeros(num_classe)
    MWCov = torch.zeros(num_classe)

    for i_sem in range(num_classe):
        MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
        MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])
    
    precision = torch.zeros(num_classe)
    recall = torch.zeros(num_classe)
    RQ = torch.zeros(num_classe)
    SQ = torch.zeros(num_classe)
    PQ = torch.zeros(num_classe)
    PQStar = torch.zeros(num_classe)
    ins_classcount = [2,3,4,6,7,8] 
    set1 = set(ins_classcount)
    set2 = set(ins_classcount_have)
    set3 = set1 & set2
    list3 = list(set3)
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
        if (tp + fp)==0:
            prec = 0
        else:
            prec = tp / (tp + fp)
        precision[i_sem] = prec
        recall[i_sem] = rec
        # RQ, SQ, PQ and PQ_star
        if (prec+rec)==0:
            RQ[i_sem] = 0
        else:
            RQ[i_sem] = 2*prec*rec/(prec+rec)
        if tp==0:
            SQ[i_sem] = 0
        else:
            SQ[i_sem] = IoU_Tp[i_sem]/tp
        PQ[i_sem] = SQ[i_sem]*RQ[i_sem]
        # PQStar[i_sem] = IoU_Mc[i_sem]/total_gt_ins[i_sem]
        PQStar[i_sem] = PQ[i_sem]
    
    if torch.mean(precision[list3])+torch.mean(recall[list3])==0:
        F1_score = torch.tensor(0.)
    else:
        F1_score = (2*torch.mean(precision[list3])*torch.mean(recall[list3]))/(torch.mean(precision[list3])+torch.mean(recall[list3]))
    cov = torch.mean(MUCov[list3])
    wcov = torch.mean(MWCov[list3])
    mPre = torch.mean(precision[list3])
    mRec = torch.mean(recall[list3])

    return cov, wcov, mPre, mRec, F1_score

def _pred_instances_per_scan(clusters, predicted_labels, scores, batch, scan_id_offset):
    # Get sample index offset
    ones = torch.ones_like(batch)
    sample_sizes = torch.cat((torch.tensor([0]).to(batch.device), scatter_add(ones, batch.long())))
    offsets = sample_sizes.cumsum(dim=-1).cpu().numpy()

    # Build instance objects
    instances = []
    for i, cl in enumerate(clusters):
        sample_idx = batch[cl[0]].item()
        scan_id = sample_idx + scan_id_offset
        indices = cl.cpu().numpy() - offsets[sample_idx]
        if scores==None:
            instances.append(
                _Instance(
                    classname=predicted_labels[cl[0]].item(), score=-1, indices=indices, scan_id=scan_id
                )
            )
        else:
            instances.append(
                _Instance(
                    classname=predicted_labels[cl[0]].item(), score=scores[i].item(), indices=indices, scan_id=scan_id
                )
            )
    return instances  
    
def _gt_instances_per_scan(instance_labels, gt_labels, batch, scan_id_offset):
    batch_size = batch[-1] + 1
    instances = []
    for b in range(batch_size):
        sample_mask = batch == b
        instances_in_sample = instance_labels[sample_mask]
        gt_labels_sample = gt_labels[sample_mask]
        num_instances = torch.max(instances_in_sample)
        scan_id = b + scan_id_offset
        for i in range(num_instances):
            instance_indices = torch.where(instances_in_sample == i + 1)[0].cpu().numpy()
            instances.append(
                _Instance(
                    classname=gt_labels_sample[instance_indices[0]].item(),
                    score=-1,
                    indices=instance_indices,
                    scan_id=scan_id,
                )
            )
    return instances

def _dict_to_str(dictionnary):
    string = "{"
    for key, value in dictionnary.items():
        string += "%s: %.2f," % (str(key), value)
    string += "}"
    return string  

data_loader = DataLoader(Dataset(), batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn,drop_last=True)
data_loader_val = DataLoader(Dataset(val=True), batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
data_loader_test = DataLoader(Dataset(test=True), batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

# a data loader must return a tuple of coords, features, and labels.
net = SetResNet()
net = net.to(device)
#wandb.watch(net)

mse = torch.nn.MSELoss()
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])


optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.0001)

for i in range(100):

    net.train()
    for step,(pos3d,coords,back_feat,sem_feat,score_gt,score_pre,group,all_clusters, gt_ins, gt_sem, batch_idx, num_ins) in enumerate(tqdm(data_loader)):
        
        optimizer.zero_grad()

        #input = ME.SparseTensor(feats, coordinates=coords, device=device)

        #label = label.to(device)

        # Forward
        cluster_scores = net(pos3d,coords,back_feat,sem_feat,score_gt,score_pre,group,all_clusters,inference=False,shuffle_groups=False)

        loss = cal_score_loss(
                score_gt,
                all_clusters,
                cluster_scores.to("cpu"),
                None,
                min_iou_threshold=args.clip_min,
                max_iou_threshold=args.clip_max,
            )

        wandb.log({'train_loss': loss.item()})

        # Gradient
        loss.backward()
        #if step%10==0:
        #    plot_grad_flow(net.named_parameters())
        wandb.log({'train_old': mse(score_gt,score_pre)})
        wandb.log({'train_mse': mse(score_gt,cluster_scores.to("cpu"))})
        optimizer.step()
        torch.cuda.empty_cache()

    #plot_grad_flow(net.named_parameters())
    net.eval()
    #preds = []
    #labels = []

    #TP = 0
    #ALL = 0
    _pos = tnt.meter.AverageValueMeter()
    _neg = tnt.meter.AverageValueMeter()
    _acc_meter = tnt.meter.AverageValueMeter()
    _cov =  tnt.meter.AverageValueMeter()
    _wcov =  tnt.meter.AverageValueMeter()
    _mIPre =  tnt.meter.AverageValueMeter()
    _mIRec =  tnt.meter.AverageValueMeter()
    _F1 =  tnt.meter.AverageValueMeter()
    _ap_meter = InstanceAPMeter()
    _scan_id_offset = 0
    preds = []
    old_preds = []
    labels = []
    with torch.no_grad():
        for pos3d,coords,back_feat,sem_feat,score_gt,score_pre,group,all_clusters,gt_ins, gt_sem, batch_idx, num_ins in tqdm(data_loader_test):
            #input = ME.SparseTensor(feats, coordinates=coords, device=device)
            # label = label.to(device)
            cluster_scores = net(pos3d,coords,back_feat,sem_feat,score_gt,score_pre,group,all_clusters,inference=True,shuffle_groups=False)
            #TP+=TP_i
            #ALL+=ALL_i

            torch.cuda.empty_cache()
            
            #NMS according to scores 
            clusters, valid_c_idx = extract_clusters(all_clusters, cluster_scores.to("cpu"), sem_feat)
            predicted_sem_labels = sem_feat.max(1)[1]
            
            if clusters:
                if torch.max(gt_ins)>0:
                    tp, fp, acc = compute_acc(
                        clusters, predicted_sem_labels, gt_ins, gt_sem, batch_idx, num_ins, 0.5
                    )
                    _pos.add(tp)
                    _neg.add(fp)
                    _acc_meter.add(acc)
                    
                    cov, wcov, mPre, mRec, F1 = compute_eval(
                        clusters, predicted_sem_labels, gt_ins, gt_sem, batch_idx, num_ins, 9, 0.5
                    )
                    _cov.add(cov)
                    _wcov.add(wcov)
                    _mIPre.add(mPre)
                    _mIRec.add(mRec)
                    _F1.add(F1)
        
                    # Track instances for AP
                    pred_clusters = _pred_instances_per_scan(
                        clusters, predicted_sem_labels, cluster_scores, batch_idx, _scan_id_offset
                    )
                    gt_clusters = _gt_instances_per_scan(
                        gt_ins, gt_sem, batch_idx, _scan_id_offset
                    )
                    _ap_meter.add(pred_clusters, gt_clusters)
                    _scan_id_offset += batch_idx[-1].item() + 1
            
            preds.append(cluster_scores.to("cpu"))
            labels.append(score_gt)
            old_preds.append(score_pre)        
        rec, _, ap = _ap_meter.eval(0.5)
        _ap = OrderedDict(sorted(ap.items()))
        _rec = OrderedDict({})
        for key, val in sorted(rec.items()):
            try:
                value = val[-1]
            except TypeError:
                value = val
            _rec[key] = value
            
        metrics = {}
        _stage='test'
        metrics["{}_pos".format(_stage)] = meter_value(_pos)
        metrics["{}_neg".format(_stage)] = meter_value(_neg)
        metrics["{}_Iacc".format(_stage)] = meter_value(_acc_meter)
        
        metrics["{}_cov".format(_stage)] = meter_value(_cov)
        metrics["{}_wcov".format(_stage)] = meter_value(_wcov)
        metrics["{}_mIPre".format(_stage)] = meter_value(_mIPre)
        metrics["{}_mIRec".format(_stage)] = meter_value(_mIRec)
        metrics["{}_F1".format(_stage)] = meter_value(_F1)

        mAP = sum(_ap.values()) / len(_ap)
        metrics["{}_map".format(_stage)] = mAP

        metrics["{}_class_rec".format(_stage)] = _dict_to_str(_rec)
        metrics["{}_class_ap".format(_stage)] = _dict_to_str(_ap)
    wandb.log(metrics)
    wandb.log({'test_mseold': mse(torch.cat(labels),torch.cat(old_preds))})
    wandb.log({'test_mse': mse(torch.cat(labels),torch.cat(preds))}) 
    print(metrics)
        #preds.extend(all_preds)
        #labels.extend(all_labels)
    #preds = np.round(preds)
    #labels = np.round(labels)
    #precision = precision_score(y_true=labels,y_pred=preds)
    #recall = recall_score(y_true=labels,y_pred=preds)
    #f1 = f1_score(y_true=labels,y_pred=preds)
    #print({'mse': mse(torch.Tensor(labels),torch.Tensor(preds)),"precision":precision,"recall":recall,"F1":f1,"actuall Acc":TP/ALL})
    #try:
    #    wandb.log({'mse': mse(torch.Tensor(labels),torch.Tensor(preds)),"precision":precision,"recall":recall,"F1":f1,"actual Acc":TP/ALL})
    #except:# if zero division error because there is no actual Acc
    #    wandb.log({'mse': mse(torch.Tensor(labels), torch.Tensor(preds)), "precision": precision, "recall": recall, "F1": f1})
