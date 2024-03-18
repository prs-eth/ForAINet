import os
import numpy as np
from plyfile import PlyData, PlyElement
from utils.utils import output_DTM_as_pc, DTM_generation, DTM_accuracy
from scipy.interpolate import RectBivariateSpline, interp2d, bisplrep, bisplev, griddata
import hdbscan
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from itertools import product
from multiprocessing import Pool
from PIL import Image
from RANSAC.RANSAC.RANSACCircle_2 import run
from joblib import Parallel, delayed
import sys
import json

def hdbscan_filtering(point_cloud, cluster_selection_epsilon):#min_cluster_size, min_samples):
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
    hdbscan_model = hdbscan.HDBSCAN(min_samples=10, cluster_selection_epsilon=cluster_selection_epsilon)#min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Fit the model to the point cloud data and get the cluster labels
    cluster_labels = hdbscan_model.fit_predict(point_cloud)

    # Find the most frequent non-negative label (excluding -1)
    non_negative_labels = cluster_labels[cluster_labels >= 0]
    most_frequent_label = np.bincount(non_negative_labels).argmax()

    # Filter the points based on the most frequent label
    filtered_points = point_cloud[cluster_labels == most_frequent_label]

    # Find the indices of the filtered points in the original point_cloud
    remain_idx = np.where(cluster_labels == most_frequent_label)[0]

    return filtered_points, remain_idx

def hdbscan_filtering_2d(point_cloud, min_points_threshold, cluster_selection_epsilon, min_cluster_size, min_samples):
    """
    HDBSCAN-based neighborhood filtering function for 2D point cloud

    Parameters:
    point_cloud: np.array, shape (N, 2), representing the input 2D point cloud data

    Returns:
    filtered_points: np.array, shape (M, 2), representing the filtered 2D point cloud data
    remain_idx: np.array, shape (M,), representing the indices of the filtered points in the original point_cloud
    """    
    # Create the HDBSCAN model with appropriate parameters
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
    
    #if no enough points for fitting
    if np.shape(filtered_points)[0] < min_points_threshold:
        return point_cloud, np.arange(np.shape(point_cloud)[0])

    return filtered_points, remain_idx

def cal_DBH_and_centerP(points_for_fitting, im_path, instanceId, min_points_threshold, cluster_selection_epsilon, min_cluster_size, min_samples):
    """
    Calculate the Diameter at Breast Height (DBH) and center point of a tree trunk from the given points.

    This function takes a set of 2D points representing the tree trunk and fits a circle to the points.
    The circle's diameter is used to calculate the Diameter at Breast Height (DBH) of the tree.
    The function also saves a visual representation of the fitted circle and the points as an image.

    Parameters:
        points_for_fitting (np.ndarray): A 2D NumPy array representing the points for fitting the circle.
                                        Each row contains the x and y coordinates of a point.
        fig (matplotlib.figure.Figure): A Matplotlib figure object where the visualizations will be displayed.
        im_path (str): The file path to save the image of the fitted circle and points.
        instanceId (int): An identifier for the tree instance.

    Returns:
        float: The calculated Diameter at Breast Height (DBH) of the tree trunk in meters.

    Note:
        The 'points_for_fitting' should be a NumPy array of shape (N, 2) where N is the number of points.
        The 'fig' parameter is a Matplotlib figure where the visualizations will be added as a subplot.
        The 'im_path' should be a string specifying the file path where the image will be saved.
        The 'instanceId' is an identifier for the tree instance and is used for plot and image names.

        The function first plots the projected points and fitting circle for the tree trunk as a subplot in 'fig'.
        Then, it calculates the Diameter at Breast Height (DBH) of the tree trunk using the fitting circle's diameter.
        The function saves a black-and-white image of the fitted circle and points, where the points are projected
        onto a 2D grid and a circle is fit to them. The saved image will be used to calculate the DBH using the RANSAC
        circle fitting algorithm.
    """
    #plot projected points and fitting circle for tree trunk
    
    # 2d point filtering: avoid fitting circle for more than one stem
    xy_tmp = np.concatenate((np.expand_dims(points_for_fitting[:,0],axis=1),np.expand_dims(points_for_fitting[:,1],axis=1)),axis=1)
    points_for_fitting, idx = hdbscan_filtering_2d(xy_tmp, min_points_threshold, cluster_selection_epsilon, min_cluster_size, min_samples)
    
    #height, width
    # Define the grid resolution and calculate the image width and height based on the points
    resolution = 0.002
    img_width =  np.ceil((np.max(points_for_fitting[:,0])-np.min(points_for_fitting[:,0]))/resolution) + 1
    img_height = np.ceil((np.max(points_for_fitting[:,1])-np.min(points_for_fitting[:,1]))/resolution) + 1
    
    # Create an image array filled with white pixels
    arr_image = 255*np.ones((int(img_width), int(img_height)))
    # Project each point onto the 2D grid and mark it with a black pixel
    for point_i in points_for_fitting:
        w_i = np.floor((point_i[0]-np.min(points_for_fitting[:,0]))/resolution)
        h_i = np.floor((point_i[1]-np.min(points_for_fitting[:,1]))/resolution)
        arr_image[int(w_i)][int(h_i)]=0
    
    # Convert the image array to a PIL image and save it
    im = Image.fromarray(arr_image).convert("L")
    im.save(im_path) 
    # Run the RANSAC circle fitting algorithm to calculate the trunk radius and center coordinates
    T_radius, center_Y, center_X = run(im_path, arr_image, 100, 3, instance_id=instanceId, resolution=resolution)
    if T_radius == -1:
        #fitting error
        return 0, 0, 0
    # Calculate the corresponding x, y coordinates of center points in the point cloud
    point_cloud_x = (center_X * resolution) + np.min(points_for_fitting[:, 0])
    point_cloud_y = (center_Y * resolution) + np.min(points_for_fitting[:, 1])
    
    # Return the calculated Diameter at Breast Height (DBH) and center coordinates of the tree trunk
    return T_radius*2, point_cloud_x, point_cloud_y

def cal_DBH_and_centerP_gt(points_for_fitting, im_path, instanceId):
    """
    Calculate the Diameter at Breast Height (DBH) and center point of a tree trunk from the given points.

    This function takes a set of 2D points representing the tree trunk and fits a circle to the points.
    The circle's diameter is used to calculate the Diameter at Breast Height (DBH) of the tree.
    The function also saves a visual representation of the fitted circle and the points as an image.

    Parameters:
        points_for_fitting (np.ndarray): A 2D NumPy array representing the points for fitting the circle.
                                        Each row contains the x and y coordinates of a point.
        fig (matplotlib.figure.Figure): A Matplotlib figure object where the visualizations will be displayed.
        im_path (str): The file path to save the image of the fitted circle and points.
        instanceId (int): An identifier for the tree instance.

    Returns:
        float: The calculated Diameter at Breast Height (DBH) of the tree trunk in meters.

    Note:
        The 'points_for_fitting' should be a NumPy array of shape (N, 2) where N is the number of points.
        The 'fig' parameter is a Matplotlib figure where the visualizations will be added as a subplot.
        The 'im_path' should be a string specifying the file path where the image will be saved.
        The 'instanceId' is an identifier for the tree instance and is used for plot and image names.

        The function first plots the projected points and fitting circle for the tree trunk as a subplot in 'fig'.
        Then, it calculates the Diameter at Breast Height (DBH) of the tree trunk using the fitting circle's diameter.
        The function saves a black-and-white image of the fitted circle and points, where the points are projected
        onto a 2D grid and a circle is fit to them. The saved image will be used to calculate the DBH using the RANSAC
        circle fitting algorithm.
    """
    #plot projected points and fitting circle for tree trunk
    #height, width
    # Define the grid resolution and calculate the image width and height based on the points
    resolution = 0.002
    img_width =  np.ceil((np.max(points_for_fitting[:,0])-np.min(points_for_fitting[:,0]))/resolution) + 1
    img_height = np.ceil((np.max(points_for_fitting[:,1])-np.min(points_for_fitting[:,1]))/resolution) + 1
    
    # Create an image array filled with white pixels
    arr_image = 255*np.ones((int(img_width), int(img_height)))
    # Project each point onto the 2D grid and mark it with a black pixel
    for point_i in points_for_fitting:
        w_i = np.floor((point_i[0]-np.min(points_for_fitting[:,0]))/resolution)
        h_i = np.floor((point_i[1]-np.min(points_for_fitting[:,1]))/resolution)
        arr_image[int(w_i)][int(h_i)]=0
    
    # Convert the image array to a PIL image and save it
    im = Image.fromarray(arr_image).convert("L")
    im.save(im_path) 
    # Run the RANSAC circle fitting algorithm to calculate the trunk radius and center coordinates
    T_radius, center_Y, center_X = run(im_path, arr_image, 100, 3, instance_id=instanceId, resolution=resolution)
    if T_radius == -1:
        #fitting error
        return 0, 0, 0
    # Calculate the corresponding x, y coordinates of center points in the point cloud
    point_cloud_x = (center_X * resolution) + np.min(points_for_fitting[:, 0])
    point_cloud_y = (center_Y * resolution) + np.min(points_for_fitting[:, 1])
    
    # Return the calculated Diameter at Breast Height (DBH) and center coordinates of the tree trunk
    return T_radius*2, point_cloud_x, point_cloud_y

file_path_out = '/scratch2/OutdoorPanopticSeg_V2/outputs/tree_set1/tree_set1-PointGroup-PAPER-20230612_095017/eval/2023-07-19_10-37-44/rmse_params.json'
file_path = '/scratch2/OutdoorPanopticSeg_V2/outputs/tree_set1/tree_set1-PointGroup-PAPER-20230612_095017/eval/2023-07-19_10-37-44/'
all_files = os.listdir(file_path)
matching_files = [filename for filename in all_files if 'Semantic_results_forEval' in filename]
directory = file_path+'para_cal_imgs'
if not os.path.exists(directory):
    os.makedirs(directory)
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
    'cluster_selection_epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'min_cluster_size': np.linspace(3, 10, 2).astype(int),  #5
    'min_samples':  np.linspace(3, 5, 1).astype(int)   #10
}
#min_cluster_size = 50
#min_samples = 10
#Iterate over all plots
best_height_epsilon = 1.0
#predefined parameters
BINSIZE = 0.5 #for dtm rasterization
height_of_trunkDiameter = 1.3
# Set the minimum points threshold
min_points_threshold = 10
def evaluate_rmse_dbh(cluster_selection_epsilon, min_cluster_size, min_samples):
    min_cluster_size = int(min_cluster_size)
    min_samples = int(min_samples)
    RMSE_Total = []
    for f_idx, filename in enumerate(matching_files):
        #predicted semantic segmentation file path
        pred_class_label_filename = file_path + filename
        new_dir_name = "plot" + pred_class_label_filename[-6:-4]
        # Construct the full path to the new directory for new plot
        current_plot = os.path.join(directory, new_dir_name)
        if not os.path.exists(current_plot):
            os.mkdir(current_plot)
        
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
        #veg_height_gt = np.zeros_like(veg_ins_gt, dtype=float)
        veg_Tdiamater_gt = np.zeros_like(veg_ins_gt, dtype=float)

        #predicted instances for vegetation
        x_veg_pre = data_ins['vertex']['x'][idx_pre_tree]
        y_veg_pre = data_ins['vertex']['y'][idx_pre_tree]
        z_veg_pre = data_ins['vertex']['z'][idx_pre_tree]
        veg_ins_pre = pred_ins_complete[idx_pre_tree]
        veg_sem_pre = pred_sem_complete[idx_pre_tree]
        #veg_height_pre = np.zeros_like(veg_ins_pre, dtype=float)
        veg_Tdiamater_pre =  np.zeros_like(veg_ins_pre, dtype=float)  #Trunk diameter
        
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
            #filtered_points, con_4 = hdbscan_filtering(xyz_tmp,best_height_epsilon)#, min_cluster_size, min_samples)
            
            #if np.size(con_4)==0:
            #    break
            
            #veg_height_pre[tmp] = np.max(filtered_points[:,2])
            
            ######################calculate DBH (diameter at breast height)        
            # Get points for fitting circle
            # Define a threshold range for selecting points based on their z-coordinates
            # We will look for points within the range [height_of_trunkDiameter - 0.5, height_of_trunkDiameter + 0.5]
            con_1 = np.where(z_tmp > (height_of_trunkDiameter - 0.5))[0]
            con_2 = np.where(z_tmp < (height_of_trunkDiameter + 0.5))[0]
            # Select points that belong to the stem class (assumed to be labeled as 2)
            con_3 = np.where(veg_sem_pre[tmp] == 2)
            
            if np.size(con_3)<min_points_threshold:
                continue
            
            # Find the indices of points that satisfy all the conditions
            idx_for_fitting = np.intersect1d(con_1, con_2)
            idx_for_fitting = np.intersect1d(idx_for_fitting, con_3)
            #idx_for_fitting = np.intersect1d(idx_for_fitting, con_4)  #will alert error for RMIT forestry area, because too sparse points for stem

            # Set the initial increment for adjusting the height range
            increment = 0.5

            # Keep increasing the height range until we have at least 20 points that satisfy the conditions
            while np.size(idx_for_fitting) < min_points_threshold and increment<4:
                # Increase the increment value for the next iteration (adjust the height range further)
                increment += 0.2
                # Update the height threshold range
                height_threshold_upper = height_of_trunkDiameter + increment
                height_threshold_lower = height_of_trunkDiameter - increment

                # Find new points that satisfy the updated height conditions
                con_1 = np.where(z_tmp > height_threshold_lower)[0]
                con_2 = np.where(z_tmp < height_threshold_upper)[0]
                con_3 = np.where(veg_sem_pre[tmp] == 2)
                idx_for_fitting = np.intersect1d(con_1, con_2)
                idx_for_fitting = np.intersect1d(idx_for_fitting, con_3)
                #idx_for_fitting = np.intersect1d(idx_for_fitting, con_4)

            if np.size(idx_for_fitting) < min_points_threshold:
                continue
            points_for_fitting = xyz_tmp[idx_for_fitting]
            #calculate DBH, center point and save figures
            im_path = current_plot+'/'+ 'preTree'+str(g)+ '_fittingPointsProj.png'
            dbh, center_x, center_y = cal_DBH_and_centerP(points_for_fitting, im_path, g, min_points_threshold, cluster_selection_epsilon, min_cluster_size, min_samples)
            veg_Tdiamater_pre[tmp] = dbh
            
            
            
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
            
            tmp_tree = np.where(veg_ins_gt == g)
            tmp_com = np.where(gt_ins_complete == g) 
            pts_in_gt += [tmp_com[0]]
            pts_in_gt_tree += [tmp_tree[0]]
            
            xyz_tmp = np.concatenate((np.concatenate((np.expand_dims(x_tmp,axis=1),np.expand_dims(y_tmp,axis=1)),axis=1), np.expand_dims(z_tmp,axis=1)), axis=1)
            #filtered_points = hdbscan_filtering(xyz_tmp, min_cluster_size, min_samples)
            
            #veg_height_gt[tmp] = np.max(z_tmp)
            
            ######################calculate DBH (diameter at breast height)        
            # Get points for fitting circle
            # Define a threshold range for selecting points based on their z-coordinates
            # We will look for points within the range [height_of_trunkDiameter - 0.2, height_of_trunkDiameter + 0.2]
            con_1 = np.where(z_tmp > (height_of_trunkDiameter - 0.5))[0]
            con_2 = np.where(z_tmp < (height_of_trunkDiameter + 0.5))[0]
            # Select points that belong to the vegetation class (assumed to be labeled as 2)
            con_3 = np.where(veg_sem_gt[tmp]==2)
            
            if np.size(con_3)<min_points_threshold:
                continue
            
            # Find the indices of points that satisfy all the conditions
            idx_for_fitting = np.intersect1d(con_1, con_2)
            idx_for_fitting = np.intersect1d(idx_for_fitting, con_3)
            #idx_for_fitting = np.intersect1d(idx_for_fitting, con_4)
            
            # Set the initial increment for adjusting the height range
            increment = 0.5

            # Keep increasing the height range until we have at least 20 points that satisfy the conditions
            while np.size(idx_for_fitting) < min_points_threshold and increment<4:
                # Increase the increment value for the next iteration (adjust the height range further)
                increment += 0.2
                
                # Update the height threshold range
                height_threshold_upper = height_of_trunkDiameter + increment
                height_threshold_lower = height_of_trunkDiameter - increment

                # Find new points that satisfy the updated height conditions
                con_1 = np.where(z_tmp > height_threshold_lower)[0]
                con_2 = np.where(z_tmp < height_threshold_upper)[0]
                con_3 = np.where(veg_sem_gt[tmp] == 2)
                idx_for_fitting = np.intersect1d(con_1, con_2)
                idx_for_fitting = np.intersect1d(idx_for_fitting, con_3)
                #idx_for_fitting = np.intersect1d(idx_for_fitting, con_4)

            if np.size(idx_for_fitting) < min_points_threshold:
                continue
            points_for_fitting = xyz_tmp[idx_for_fitting]
            #calculate DBH, center point and save figures
            im_path = current_plot+'/'+ 'gtTree'+str(g)+ '_fittingPointsProj.png'
            dbh, center_x, center_y = cal_DBH_and_centerP_gt(points_for_fitting, im_path, g)
            
            veg_Tdiamater_gt[tmp] = dbh #Trunk diameter
            
        
        error_TD = []
        
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
                error_TD += [veg_Tdiamater_pre[cor_pre][0]-veg_Tdiamater_gt[cor_gt][0]]
        
        RMSE_TD = np.sqrt(np.mean(np.square(error_TD)))
        RMSE_Total.append(RMSE_TD)
    rmse_f = np.mean(np.array(RMSE_Total))
    rmse_params = {
        'rmse_f': rmse_f,
        'cluster_selection_epsilon': cluster_selection_epsilon,
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples
    }
    print(rmse_params)
    
    # Check if the file exists
    if os.path.exists(file_path_out):
        # File exists, open in append mode
        with open(file_path_out, 'a') as json_file:
            json.dump(rmse_params, json_file)
            json_file.write('\n')  # Add a newline to separate the dictionaries (optional)
    else:
        # File doesn't exist, create and write the first dictionary
        with open(file_path_out, 'w') as json_file:
            json.dump(rmse_params, json_file)

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
    #cluster_selection_epsilon = params[0]
    #return evaluation_function(cluster_selection_epsilon)
    cluster_selection_epsilon, min_cluster_size, min_samples = params
    return evaluation_function(cluster_selection_epsilon, min_cluster_size, min_samples)

def parallel_grid_search(param_grid, evaluation_function, num_workers=None):
    if num_workers is None:
        num_workers = os.cpu_count()

    param_combinations = list(product(*param_grid.values()))


    # Redirect stdout and stderr to devnull (hide output)
    #sys.stdout = open(os.devnull, 'w')
    #sys.stderr = open(os.devnull, 'w')
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(parallel_evaluation, [(params, evaluation_function) for params in param_combinations])
    #results = Parallel(n_jobs=num_workers)(delayed(parallel_evaluation)(params, evaluation_function) for params in param_combinations)
    # Restore stdout and stderr
    #sys.stdout = sys.__stdout__
    #sys.stderr = sys.__stderr__

    best_idx = np.argmin(results)
    best_params = dict(zip(param_grid.keys(), param_combinations[best_idx]))
    best_rmse = results[best_idx]

    return best_params, best_rmse
# Example usage:
best_params, best_rmse = parallel_grid_search(param_grid, evaluate_rmse_dbh)
print("Best parameters:", best_params)
print("Best RMSE:", best_rmse)
with open(file_path_out, 'a') as json_file:
    json.dump(best_params, json_file)
