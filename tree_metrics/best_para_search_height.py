import os
import numpy as np
from plyfile import PlyData, PlyElement
from utils.utils import output_DTM_as_pc, DTM_generation, cal_DBH_and_centerP, DTM_accuracy
from scipy.interpolate import RectBivariateSpline, interp2d, bisplrep, bisplev, griddata
import hdbscan
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from itertools import product
from multiprocessing import Pool
from joblib import parallel_backend
from joblib import Parallel, delayed

def hdbscan_filtering(point_cloud, cluster_selection_epsilon, min_cluster_size, min_samples):
    """
    HDBSCAN-based neighborhood filtering function

    Parameters:
    point_cloud: np.array, shape (N, 3), representing the input point cloud data

    Returns:
    filtered_points: np.array, shape (M, 3), representing the filtered point cloud data
    remain_idx: np.array, shape (M,), representing the indices of the filtered points in the original point_cloud
    """
    # Create the HDBSCAN model
    cluster_selection_epsilon = float(cluster_selection_epsilon)
    hdbscan_model = hdbscan.HDBSCAN(cluster_selection_epsilon=cluster_selection_epsilon, min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Fit the model to the point cloud data and get the cluster labels
    cluster_labels = hdbscan_model.fit_predict(point_cloud)

    # Find the most frequent non-negative label (excluding -1)
    non_negative_labels = cluster_labels[cluster_labels >= 0]
    if np.size(non_negative_labels)==0:
        return point_cloud, np.arange(np.shape(point_cloud)[0])
    most_frequent_label = np.bincount(non_negative_labels).argmax()

    # Filter the points based on the most frequent label
    filtered_points = point_cloud[cluster_labels == most_frequent_label]

    # Find the indices of the filtered points in the original point_cloud
    remain_idx = np.where(cluster_labels == most_frequent_label)[0]

    return filtered_points, remain_idx

file_path = '/scratch2/OutdoorPanopticSeg_V2/outputs/tree_set1/tree_set1-PointGroup-PAPER-20230612_095017/eval/2023-07-19_10-37-44/'
all_files = os.listdir(file_path)
matching_files = [filename for filename in all_files if 'Semantic_results_forEval' in filename]
#predefined parameters
BINSIZE = 0.5 #for dtm rasterization
data_class_t = []
data_ins_t = []
dtm_gt_t = []
dtm_pre_t = []
for f_idx, filename in enumerate(matching_files):
    #predicted semantic segmentation file path
    pred_class_label_filename = file_path + filename
    #predicted instance segmentation file path
    pred_ins_label_filename = file_path +filename.replace('Semantic_results_forEval', 'Instance_Results_forEval')

    #read files
    data_class = PlyData.read(pred_class_label_filename) 
    data_class_t.append(data_class)
    data_ins = PlyData.read(pred_ins_label_filename)
    data_ins_t.append(data_ins)
    pred_ins_complete = data_ins['vertex']['preds']
    pred_sem_complete = data_class['vertex']['preds']
    gt_ins_complete = data_ins['vertex']['gt'] -1
    gt_sem_complete = data_class['vertex']['gt']

    #idx for ground points
    idx_gt_ground = (gt_sem_complete==1)
    idx_pre_ground = (pred_sem_complete==1)

    ##########################Attributes for plot-level###########################
    #DTM generation from ground points
    #DTM of gt
    dtm_gt = DTM_generation(data_ins['vertex']['x'][idx_gt_ground], data_ins['vertex']['y'][idx_gt_ground], data_ins['vertex']['z'][idx_gt_ground], BINSIZE)
    dtm_gt_t.append(dtm_gt)
    #DTM of prediction
    dtm_pre = DTM_generation(data_ins['vertex']['x'][idx_pre_ground], data_ins['vertex']['y'][idx_pre_ground], data_ins['vertex']['z'][idx_pre_ground], BINSIZE)
    dtm_pre_t.append(dtm_pre)
    #output DTM as point cloud
    
# Define HDBSCAN parameters

# Define the parameter grid
#param_grid = {
#    'min_cluster_size': np.linspace(100, 500, 100).astype(int),
#    'min_samples':  np.linspace(10, 100, 10).astype(int)
#}
param_grid = {
    'cluster_selection_epsilon': [0.1, 0.2, 0.3, 0.4, 0.5], #np.arange(0.5, 3, 0.2),   #best:0.5
    'min_cluster_size': [40, 50, 60, 70, 80],  #np.linspace(5, 55, 10).astype(int),  #5  #best:43
    'min_samples':  [5] #np.linspace(5, 30, 5).astype(int)   #10  #best:5
}
#min_cluster_size = 50
#min_samples = 10
#Iterate over all plots
def evaluate_rmse_height(cluster_selection_epsilon, min_cluster_size, min_samples):
    min_cluster_size = int(min_cluster_size)
    min_samples = int(min_samples)
    RMSE_Total = []
    for f_idx, filename in enumerate(matching_files):
        data_class = data_class_t[f_idx]
        data_ins = data_ins_t[f_idx]
        pred_ins_complete = data_ins['vertex']['preds']
        pred_sem_complete = data_class['vertex']['preds']
        gt_ins_complete = data_ins['vertex']['gt'] -1
        gt_sem_complete = data_class['vertex']['gt']

        ##########################Attributes for plot-level###########################
        #DTM generation from ground points
        #DTM of gt
        dtm_gt = dtm_gt_t[f_idx]
        #DTM of prediction
        dtm_pre = dtm_pre_t[f_idx]
        #output DTM as point cloud
        
        #Stem density for classify complexity of the plot
        #idx for vegetation points
        idx_gt_tree = (gt_sem_complete==2)|(gt_sem_complete==3)|(gt_sem_complete==4)
        idx_pre_tree = (pred_sem_complete==2)|(pred_sem_complete==3)|(pred_sem_complete==4)

        #GT instances for vegetation
        x_veg_gt = data_ins['vertex']['x'][idx_gt_tree]
        y_veg_gt = data_ins['vertex']['y'][idx_gt_tree]
        z_veg_gt = data_ins['vertex']['z'][idx_gt_tree]
        veg_ins_gt = gt_ins_complete[idx_gt_tree]
        veg_sem_gt = gt_sem_complete[idx_gt_tree]
        cor_veg_ins_pre = -1*np.ones_like(veg_ins_gt)
        veg_height_gt = np.zeros_like(veg_ins_gt, dtype=float)

        #predicted instances for vegetation
        x_veg_pre = data_ins['vertex']['x'][idx_pre_tree]
        y_veg_pre = data_ins['vertex']['y'][idx_pre_tree]
        z_veg_pre = data_ins['vertex']['z'][idx_pre_tree]
        veg_ins_pre = pred_ins_complete[idx_pre_tree]
        veg_sem_pre = pred_sem_complete[idx_pre_tree]
        veg_height_pre = np.zeros_like(veg_ins_pre, dtype=float)
        
        #tree instances in prediction set
        #print(fig.dpi)  #dpi=100
        un = np.unique(veg_ins_pre)
        pts_in_pred = []
        pts_in_pred_tree = []
        #spline of ground
        interp_spline = bisplrep(dtm_pre[:,0], dtm_pre[:,1], dtm_pre[:,2])
        ##########################Attributes for tree-level###########################
        #loop for each predicted tree instance
        for ig, g in enumerate(un):
            if g == -1:
                continue
            tmp = np.where(veg_ins_pre == g)
            x_tmp = x_veg_pre[tmp]
            y_tmp = y_veg_pre[tmp]
            z_tmp = np.copy(z_veg_pre[tmp])
            
            #get ground height based on DTM
            x_center = np.mean(x_tmp)
            y_center = np.mean(y_tmp)
            ground_z = bisplev(x_center, y_center, interp_spline)
            #ground_z = griddata((dtm_pre[:,0], dtm_pre[:,1]), dtm_pre[:,2], (x_tmp, y_tmp), method='linear')
            #z value above ground
            z_tmp = z_tmp-ground_z
            
            tmp_tree = np.where(veg_ins_pre == g)
            tmp_com = np.where(pred_ins_complete == g)
            pts_in_pred += [tmp_com[0]]
            pts_in_pred_tree += [tmp_tree[0]]
            
            xyz_tmp = np.concatenate((np.concatenate((np.expand_dims(x_tmp,axis=1),np.expand_dims(y_tmp,axis=1)),axis=1), np.expand_dims(z_tmp,axis=1)), axis=1)
            filtered_points, con_4 = hdbscan_filtering(xyz_tmp,cluster_selection_epsilon, min_cluster_size, min_samples)
            
            if np.size(con_4)==0:
                break
            
            veg_height_pre[tmp] = np.max(filtered_points[:,2])
              
        #tree instances in GT set
        #print(fig.dpi)  #dpi=100
        un = np.unique(veg_ins_gt)
        pts_in_gt = []
        pts_in_gt_tree = []
        interp_spline = bisplrep(dtm_gt[:,0], dtm_gt[:,1], dtm_gt[:,2])
        for ig, g in enumerate(un):
            if g == -1:
                continue
            tmp = np.where(veg_ins_gt == g)
            x_tmp = x_veg_gt[tmp]
            y_tmp = y_veg_gt[tmp]
            z_tmp = np.copy(z_veg_gt[tmp])
            
            #get ground height based on DTM
            x_center = np.mean(x_tmp)
            y_center = np.mean(y_tmp)
            ground_z = bisplev(x_center, y_center, interp_spline)
            #ground_z = griddata((dtm_pre[:,0], dtm_pre[:,1]), dtm_pre[:,2], (x_tmp, y_tmp), method='linear')
            #z value above ground
            z_tmp = z_tmp-ground_z
            
            xyz_tmp = np.concatenate((np.concatenate((np.expand_dims(x_tmp,axis=1),np.expand_dims(y_tmp,axis=1)),axis=1), np.expand_dims(z_tmp,axis=1)), axis=1)
            #filtered_points = hdbscan_filtering(xyz_tmp, min_cluster_size, min_samples)
            
            veg_height_gt[tmp] = np.max(z_tmp)
            
            tmp_tree = np.where(veg_ins_gt == g)
            tmp_com = np.where(gt_ins_complete == g) 
            pts_in_gt += [tmp_com[0]]
            pts_in_gt_tree += [tmp_tree[0]]
        
        gt_H = []
        pre_H = []
        error_H = []
        
        at=0.5
        # loop for each GT instance
        for ig, ins_gt in enumerate(pts_in_gt):
            ovmax = 0
            # find the best matched prediction
            for ip, ins_pred in enumerate(pts_in_pred):
                union = len(list(set(ins_pred).union(set(ins_gt)))) 
                intersect = len(list(set(ins_pred).intersection(set(ins_gt)))) 
                iou = float(intersect) / union 

                if iou > ovmax:
                    ovmax = iou
                    ipmax = ip

            if ovmax >= at:
                cor_gt = pts_in_gt_tree[ig]
                cor_pre = pts_in_pred_tree[ipmax]
                cor_veg_ins_pre[cor_gt] = pred_ins_complete[pts_in_pred[ipmax][0]]
                error_H += [veg_height_pre[cor_pre][0]-veg_height_gt[cor_gt][0]]
        
        RMSE_H = np.sqrt(np.mean(np.square(error_H)))
        RMSE_Total.append(RMSE_H)
    rmse_f = np.mean(np.array(RMSE_Total))
    rmse_params = {
        'rmse_f': rmse_f,
        'cluster_selection_epsilon': cluster_selection_epsilon,
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples
    }
    print(rmse_params)
    return rmse_f


# Custom GridSearch function
'''
def grid_search(param_grid, evaluation_function):
    best_params = None
    best_rmse = float('inf')

    # Iterate through all parameter combinations
    for params in ParameterGrid(param_grid):
        # Call the evaluation function with the current parameter combination
        rmse = evaluation_function(**params)

        # Check if current RMSE is better than the previous best RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return best_params, best_rmse

# Perform the grid search
best_params, best_rmse = grid_search(param_grid, evaluate_rmse_height)

print("Best parameters:", best_params)
print("Best RMSE:", best_rmse)
'''

def parallel_evaluation(params, evaluation_function):
    #min_cluster_size, min_samples = params
    #return evaluation_function(min_cluster_size, min_samples)
    cluster_selection_epsilon, min_cluster_size, min_samples = params
    #cluster_selection_epsilon = params[0]
    return evaluation_function(cluster_selection_epsilon, min_cluster_size, min_samples)

def parallel_grid_search(param_grid, evaluation_function, num_workers=None):
    if num_workers is None:
        num_workers = os.cpu_count()

    param_combinations = list(product(*param_grid.values()))

    #with parallel_backend('loky', n_jobs=num_workers):
    #    with Pool(processes=num_workers) as pool:
    #        results = pool.starmap(parallel_evaluation, [(params, evaluation_function) for params in param_combinations])
    results = Parallel(n_jobs=num_workers)(delayed(parallel_evaluation)(params, evaluation_function) for params in param_combinations)

    best_idx = np.argmin(results)
    best_params = dict(zip(param_grid.keys(), param_combinations[best_idx]))
    best_rmse = results[best_idx]

    return best_params, best_rmse
# Example usage:
best_params, best_rmse = parallel_grid_search(param_grid, evaluate_rmse_height)
print("Best parameters:", best_params)
print("Best RMSE:", best_rmse)
