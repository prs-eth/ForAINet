import numpy as np
from plyfile import PlyData, PlyElement
from pylidar.toolbox import spatial
from pylidar.toolbox import interpolation
from PIL import Image
from RANSAC.RANSAC.RANSACCircle_2 import run
import matplotlib.pyplot as plt
from tree_metrics.rmse import rmse
import hdbscan
from utils.ply import read_ply, write_ply
from scipy.spatial import cKDTree
from scipy.sparse import csgraph
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MeanShift
from scipy.sparse import coo_matrix
import alphashape
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

def output_DTM_as_pc(dtm, filepath):
    """
    Save a Digital Terrain Model (DTM) as a point cloud file in PLY format.

    Parameters:
        dtm (np.ndarray): A 2D NumPy array representing the DTM.
                          The array should have shape (-1, 3), where each row contains the x, y, and z
                          coordinates of a point in the DTM.
        filepath (str): The file path where the PLY file will be saved.

    Returns:
        None

    Note:
        This function saves the DTM as a point cloud in PLY format, which is a simple file format for
        storing 3D point cloud data. The input DTM should be a 2D NumPy array with the first three columns
        representing the x, y, and z coordinates of each point in the DTM. The PLY file will contain a
        'vertex' element with three properties: 'x', 'y', and 'z', each storing the respective coordinate
        values. The file will be saved to the specified 'filepath'.
    """
    body = [dtm[:,0].reshape(-1), dtm[:,1].reshape(-1), dtm[:,2].reshape(-1)]
    names = "x, y, z"
    formats = "f4, f4, f4"
    vertices = np.core.records.fromarrays(
        np.dstack(body).reshape((-1, len(body))).transpose(),
        names=names,
        formats=formats).flatten()

    ply_data_elm = [
        PlyElement.describe(
            vertices, "vertex", comments=["vertices"])
    ]
    ply_data = PlyData(ply_data_elm)
    pc_out_dest = filepath
    ply_data.write(pc_out_dest)

def DTM_generation(floor_point_x, floor_point_y, floor_point_z, binsize):
    """
    Generate a Digital Terrain Model (DTM) from floor point data using interpolation.

    Parameters:
        floor_point_x (array-like): Array of x-coordinates of floor points.
        floor_point_y (array-like): Array of y-coordinates of floor points.
        floor_point_z (array-like): Array of z-coordinates of floor points.
        binsize (float): Size of the bins used for grid generation.

    Returns:
        np.ndarray: A 2D array containing the pixel coordinates and corresponding interpolated DTM values.
                    The array has shape (-1, 3), where each row represents a valid point in the DTM, and
                    the three columns are x-coordinate, y-coordinate, and DTM value respectively.
    """
    
    # Get grid information (xmin, ymax, ncols, nrows) from the floor point data and binsize
    (xMin, yMax, ncols, nrows) = spatial.getGridInfoFromData(floor_point_x, floor_point_y,binsize)
    # Generate pixel coordinates based on the grid information and binsize
    pxlCoords = spatial.getBlockCoordArrays(xMin, yMax, ncols, nrows, binsize)
    # Generate a DTM (Digital Terrain Model) of prediction using interpolation
    # Interpolate the floor point data (x, y, z) onto the pixel coordinates using a specified interpolation method ('pynn' in this case)
    dtm_pre_valid = interpolation.interpGrid(floor_point_x, floor_point_y, floor_point_z, pxlCoords, method='pynn')
    # Identify valid points in the DTM (non-NaN values) and filter them
    idx_valid = ~(np.isnan(dtm_pre_valid))
    body = [pxlCoords[0][idx_valid].reshape(-1), pxlCoords[1][idx_valid].reshape(-1), dtm_pre_valid[idx_valid].reshape(-1)]
    # Stack the filtered pixel coordinates and interpolated DTM values into a single array and return it
    # The resulting array will have the shape (-1, len(body)), where len(body) is the number of columns in the array (3 in this case)
    return np.dstack(body).reshape((-1, len(body)))

def DTM_accuracy(all_points, binsize, idx_gt_ground, idx_pre_ground):
    floor_point_x_gt = all_points['x'][idx_gt_ground]
    floor_point_y_gt = all_points['y'][idx_gt_ground]
    floor_point_z_gt = all_points['z'][idx_gt_ground]
    
    # Get grid information (xmin, ymax, ncols, nrows) from the floor point data and binsize
    (xMin, yMax, ncols, nrows) = spatial.getGridInfoFromData(floor_point_x_gt, floor_point_y_gt,binsize)
    # Generate pixel coordinates based on the grid information and binsize
    pxlCoords = spatial.getBlockCoordArrays(xMin, yMax, ncols, nrows, binsize)
    # Generate a DTM (Digital Terrain Model) of prediction using interpolation
    # Interpolate the floor point data (x, y, z) onto the pixel coordinates using a specified interpolation method ('pynn' in this case)
    dtm_gt_valid = interpolation.interpGrid(floor_point_x_gt, floor_point_y_gt, floor_point_z_gt, pxlCoords, method='pynn')
    idx_valid = ~(np.isnan(dtm_gt_valid))
    body = [pxlCoords[0][idx_valid].reshape(-1), pxlCoords[1][idx_valid].reshape(-1), dtm_gt_valid[idx_valid].reshape(-1)]
    DTM_gt = np.dstack(body).reshape((-1, len(body)))
    
    floor_point_x_pre = all_points['x'][idx_pre_ground]
    floor_point_y_pre = all_points['y'][idx_pre_ground]
    floor_point_z_pre = all_points['z'][idx_pre_ground]
    dtm_pre_valid = interpolation.interpGrid(floor_point_x_pre, floor_point_y_pre, floor_point_z_pre, pxlCoords, method='pynn')
    body = [pxlCoords[0][idx_valid].reshape(-1), pxlCoords[1][idx_valid].reshape(-1), dtm_pre_valid[idx_valid].reshape(-1)]
    DTM_pre = np.dstack(body).reshape((-1, len(body)))
    
    #calculate the percentage of the reference DTM covered by partners’ results
    covered = ~(np.isnan(DTM_pre[:,2]))
    
    coverage = (np.sum(covered)/np.size(DTM_gt[:,2]))*100
    #coverage = (np.size(covered)/np.size(DTM_gt[:,2]))*100
    
    #calculate RMSE
    rmse_v = rmse(DTM_gt[covered][:,-1], DTM_pre[covered][:,-1])
    
    return coverage, rmse_v

def hdbscan_filtering_2d(point_cloud, min_points_threshold):
    """
    HDBSCAN-based neighborhood filtering function for 2D point cloud

    Parameters:
    point_cloud: np.array, shape (N, 2), representing the input 2D point cloud data

    Returns:
    filtered_points: np.array, shape (M, 2), representing the filtered 2D point cloud data
    remain_idx: np.array, shape (M,), representing the indices of the filtered points in the original point_cloud
    """    
    # Create the HDBSCAN model with appropriate parameters
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3, cluster_selection_epsilon=0.1)
    #hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.1)

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
    if np.shape(filtered_points)[0] <3:#< min_points_threshold:
        return point_cloud, np.arange(np.shape(point_cloud)[0])

    return filtered_points, remain_idx
 
def cal_DBH_and_centerP(points_for_fitting, fig, im_path, instanceId, min_points_threshold):
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
    # Create a subplot to plot projected points and fitting circle for the tree trunk
    ax2 = fig.add_subplot(2, 4, 3)
    ax2.scatter(points_for_fitting[:,0], points_for_fitting[:,1], marker='.', s=0.1)
    ax2.set_aspect(1)
    
    # 2d point filtering: avoid fitting circle for more than one stem
    xy_tmp = np.concatenate((np.expand_dims(points_for_fitting[:,0],axis=1),np.expand_dims(points_for_fitting[:,1],axis=1)),axis=1)
    points_for_fitting, idx = hdbscan_filtering_2d(xy_tmp, min_points_threshold)
    ax2 = fig.add_subplot(2, 4, 4)
    ax2.scatter(points_for_fitting[:,0], points_for_fitting[:,1], marker='.', s=0.1)
    ax2.set_aspect(1)
    
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
    #T_radius, center_Y, center_X = run(im_path, arr_image, 100, 1, instance_id=instanceId, resolution=resolution)
    if T_radius == -1:
        #fitting error
        return 0, 0, 0
    # Calculate the corresponding x, y coordinates of center points in the point cloud
    point_cloud_x = (center_X * resolution) + np.min(points_for_fitting[:, 0])
    point_cloud_y = (center_Y * resolution) + np.min(points_for_fitting[:, 1])
    ax2.scatter(point_cloud_x, point_cloud_y, marker='*', s=0.5)
    
    # Set the subplot's title to display the calculated trunk radius
    ax2.set_title("trunk radius = %1.3f m" % T_radius)
    # Pause for a short time to display the figure
    plt.pause(1)
    
    # Return the calculated Diameter at Breast Height (DBH) and center coordinates of the tree trunk
    return T_radius*2, point_cloud_x, point_cloud_y

def hdbscan_filtering_old(point_cloud, min_cluster_size, th_cluster):
    """
    HDBSCAN-based neighborhood filtering function

    Parameters:
    point_cloud: np.array, shape (N, 3), representing the input point cloud data
    min_cluster_size: int, the minimum number of points required to form a cluster
    min_samples: int, the number of neighbors a point must have to be considered as a core point

    Returns:
    filtered_points: np.array, shape (M, 3), representing the filtered point cloud data
    """
    # Create the HDBSCAN model
    hdbscan_model = hdbscan.HDBSCAN()#min_cluster_size=min_cluster_size) #, min_samples=min_samples)
    
    # Fit the model to the point cloud data and get the cluster labels
    cluster_labels = hdbscan_model.fit_predict(point_cloud)

    # Filter points belonging to valid clusters (cluster labels >= 0)
    filtered_points = point_cloud[cluster_labels >= 0]
    remain_idx = np.where(cluster_labels >= 0)[0]
    
    #remove_clusters_with_few_elements
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    clusters_to_keep = unique_labels[label_counts >= th_cluster]
    filtered_cluster_labels = np.where(np.isin(cluster_labels, clusters_to_keep), cluster_labels, -1)
    filtered_points = point_cloud[filtered_cluster_labels >= 0]
    remain_idx = np.where(filtered_cluster_labels >= 0)[0]
    
    
    #output_path = '/scratch2/OutdoorPanopticSeg_V2/outputs/tree_set1/tree_set1-PointGroup-PAPER-20230612_095017/eval/2023-07-15_12-12-20/para_cal_imgs/test.ply'
    #write_ply(output_path,
    #        [point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], filtered_cluster_labels.astype('int32')],
    #        ['x', 'y', 'z', 'pre_label'])

    return filtered_points, remain_idx

def hdbscan_filtering(point_cloud):
    """
    HDBSCAN-based neighborhood filtering function

    Parameters:
    point_cloud: np.array, shape (N, 3), representing the input point cloud data

    Returns:
    filtered_points: np.array, shape (M, 3), representing the filtered point cloud data
    remain_idx: np.array, shape (M,), representing the indices of the filtered points in the original point_cloud
    """
    #for small tree, especially for RMIT region
    if np.shape(point_cloud)[0]<1000:
        return point_cloud, np.arange(np.shape(point_cloud)[0])
    
    # Create the HDBSCAN model
    # adjust cluster_selection_epsilon, min_samples and min_cluster_size for your dataset
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=5, cluster_selection_epsilon=0.2)
    
    # Fit the model to the point cloud data and get the cluster labels
    cluster_labels = hdbscan_model.fit_predict(point_cloud)

    # Find the most frequent non-negative label (excluding -1)
    non_negative_labels = cluster_labels[cluster_labels >= 0]
    most_frequent_label = np.bincount(non_negative_labels).argmax()

    # Filter the points based on the most frequent label
    filtered_points = point_cloud[cluster_labels == most_frequent_label]

    # Find the indices of the filtered points in the original point_cloud
    remain_idx = np.where(cluster_labels == most_frequent_label)[0]
    
    #output_path = '/scratch2/OutdoorPanopticSeg_V2/outputs/tree_set1/tree_set1-PointGroup-PAPER-20230612_095017/eval/2023-07-15_12-12-20/para_cal_imgs/test.ply'
    #write_ply(output_path,
    #        [point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], cluster_labels.astype('int32')],
    #        ['x', 'y', 'z', 'pre_label'])

    return filtered_points, remain_idx

def mean_shift_filtering(point_cloud, bandwidth):
    """
    Mean Shift-based neighborhood filtering function

    Parameters:
    point_cloud: np.array, shape (N, 3), representing the input point cloud data
    bandwidth: float, the bandwidth parameter for Mean Shift clustering

    Returns:
    filtered_points: np.array, shape (M, 3), representing the filtered point cloud data
    remain_idx: np.array, shape (M,), representing the indices of the filtered points in the original point_cloud
    """
    # Create the Mean Shift model
    meanshift_model = MeanShift(bandwidth=bandwidth)

    # Fit the model to the point cloud data and get the cluster labels
    cluster_labels = meanshift_model.fit_predict(point_cloud)

    # Find the most frequent non-negative label (excluding -1)
    non_negative_labels = cluster_labels[cluster_labels >= 0]
    most_frequent_label = np.bincount(non_negative_labels).argmax()

    # Filter the points based on the most frequent label
    filtered_points = point_cloud[cluster_labels == most_frequent_label]

    # Find the indices of the filtered points in the original point_cloud
    remain_idx = np.where(cluster_labels == most_frequent_label)[0]
    
    output_path = '/scratch2/OutdoorPanopticSeg_V2/outputs/tree_set1/tree_set1-PointGroup-PAPER-20230612_095017/eval/2023-07-15_12-12-20/para_cal_imgs/test.ply'
    write_ply(output_path,
            [point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], cluster_labels.astype('int32')],
            ['x', 'y', 'z', 'pre_label'])

    return filtered_points, remain_idx

def preprocess_point_cloud(point_cloud, distance_threshold):
    # Build a KD-tree for efficient nearest neighbor search
    kdtree = cKDTree(point_cloud)

    # Find neighbors within the distance_threshold for each point
    neighbors_list = kdtree.query_ball_point(point_cloud, distance_threshold)

    # Flatten the list of neighbors to get pairs of connected points
    pairs = [(i, j) for i, neighbors in enumerate(neighbors_list) for j in neighbors]

    # Create the adjacency matrix in sparse format
    N = len(point_cloud)
    adjacency_matrix_sparse = coo_matrix((np.ones(len(pairs) * 2), (np.array(pairs).flatten(), np.array(pairs).T.flatten())), shape=(N, N))
    adjacency_matrix_sparse += adjacency_matrix_sparse.T
    adjacency_matrix_sparse = (adjacency_matrix_sparse > 0).astype(int)

    # Use connected components to find the clusters
    _, cluster_labels = connected_components(csgraph=adjacency_matrix_sparse)

    # Get the unique cluster labels and their counts
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

    # Assign separate labels to points in disconnected clusters
    cluster_label_mapping = {}
    new_label = 0
    for label in unique_labels:
        mask = cluster_labels == label
        sub_cluster_indices = np.where(mask)[0]
        sub_cluster_point_cloud = point_cloud[sub_cluster_indices]

        sub_kdtree = cKDTree(sub_cluster_point_cloud)
        sub_neighbors_list = sub_kdtree.query_ball_point(sub_cluster_point_cloud, distance_threshold)

        for i, neighbors in enumerate(sub_neighbors_list):
            sub_cluster_label = new_label
            for neighbor in neighbors:
                neighbor_idx = sub_cluster_indices[neighbor]
                if neighbor_idx in cluster_label_mapping:
                    sub_cluster_label = cluster_label_mapping[neighbor_idx]
                    break

            for idx in sub_cluster_indices[neighbors]:
                cluster_label_mapping[idx] = sub_cluster_label

            if sub_cluster_label == new_label:
                new_label += 1

    # Create the final cluster labels based on the mapping
    cluster_labels_final = np.array([cluster_label_mapping[i] for i in range(N)])

    # Find the most frequent non-negative label (excluding the noise label -1)
    non_negative_labels = cluster_labels_final[cluster_labels_final >= 0]
    most_frequent_label = np.bincount(non_negative_labels).argmax()

    # Create a mask to identify points that belong to the largest cluster
    mask = cluster_labels_final == most_frequent_label

    # Filter the point_cloud based on the mask
    filtered_point_cloud = point_cloud[mask]
    remain_idx = np.where(mask)[0]

    output_path = '/scratch2/OutdoorPanopticSeg_V2/outputs/tree_set1/tree_set1-PointGroup-PAPER-20230612_095017/eval/2023-07-15_12-12-20/para_cal_imgs/test.ply'
    write_ply(output_path,
              [point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], mask.astype('int32')],
              ['x', 'y', 'z', 'pre_label'])

    return filtered_point_cloud, remain_idx

'''def alpha_shape_volume(points, alpha):
    """Compute the volume of the alpha shape."""
    alpha_shape = alphashape.alphashape(points, alpha)
    triangles = np.array(alpha_shape.faces)
    
    volume = np.sum([tetrahedron_volume_from_origin(points[triangle[0]], points[triangle[1]], points[triangle[2]]) for triangle in triangles])
    return volume, alpha_shape'''

from shapely.geometry import GeometryCollection, Polygon, LineString

def alpha_shape_volume(points, alpha):
    """Compute the volume of the alpha shape."""
    alpha_shape = alphashape.alphashape(points, alpha)

    if isinstance(alpha_shape, GeometryCollection):
        if len(alpha_shape.geoms) == 0:  # Check if it's empty
            #raise ValueError("Alpha shape resulted in an empty GeometryCollection. Try a different alpha value.")
            alpha_shape = LineString()
            return 0, alpha_shape
        max_volume = 0
        final_alpha_shape = alpha_shape.geoms[0]
        for geom in alpha_shape.geoms:
            #if isinstance(geom, Trimesh):  
            triangles = np.array(geom.faces)
            volume = np.sum([tetrahedron_volume_from_origin(points[triangle[0]], points[triangle[1]], points[triangle[2]]) for triangle in triangles])
                
            if volume > max_volume:
                max_volume = volume
                final_alpha_shape = geom
        return max_volume, final_alpha_shape
    elif isinstance(alpha_shape, LineString):
        # Handle the LineString case, maybe skip or alert the user.
        return 0, alpha_shape
    # Continue processing as normal.
    # if not GeometryCollection
    else:
        triangles = np.array(alpha_shape.faces)
        volume = np.sum([tetrahedron_volume_from_origin(points[triangle[0]], points[triangle[1]], points[triangle[2]]) for triangle in triangles])
        return volume, alpha_shape


def tetrahedron_volume_from_origin(a, b, c):
    """Compute the volume of the tetrahedron formed by the origin and the triangle vertices."""
    return abs(np.dot(a, np.cross(b, c))) / 6.0

def plot_alpha_shape(points, alpha_shape, ax):
    """Plot 3D Alpha Shape on given ax."""
    if  isinstance(alpha_shape, LineString):
        pass
    else:
        edges = alpha_shape.edges
        edge_points = [points[edge] for edge in edges]
        collection = Poly3DCollection(edge_points, alpha=0.25, facecolor='b', linewidths=0.5, edgecolors='r')
        ax.add_collection3d(collection)

import open3d as o3d       

## [Open3D ERROR] [CreateFromPointCloudAlphaShape] invalid tetra in TetraMesh
def compute_volume_with_open3d(points, file_path, alpha=10):
    
    point_cloud_path = file_path.replace('fittingPointsProj.png', 'volume_pc.ply')
    mesh_path = file_path.replace('fittingPointsProj.png', 'volume_mesh.ply')
    
    pcd = o3d.geometry.PointCloud()
    
    #HDBSCAN filtering
    filtered_points, con_4 = hdbscan_filtering(points)
    
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # estimate normals
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Create a surface mesh using alpha shape
    # [Open3D ERROR] The mesh is not watertight, and the volume cannot be computed.
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    # Create a surface mesh using Poisson surface reconstruction
    #mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)


    # Visualization
    o3d.visualization.draw_geometries([mesh])
    
    o3d.io.write_point_cloud(point_cloud_path, pcd)
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    # Estimate volume
    total_volume = mesh.get_volume()
    
    return total_volume

def compute_volume_with_voxelization(points, file_path, voxel_size=0.1):
    """
    Compute volume of a point cloud using voxelization.

    Parameters:
    - points: numpy array of point cloud data.
    - file_path: path to save the visualization files.
    - voxel_size: the size of the voxel. Defaults to 0.1.

    Returns:
    - Total volume in cubic meters.
    """
    # Replace the file path to save the point cloud and voxel visualization
    point_cloud_path = file_path.replace('fittingPointsProj.png', 'volume_pc.ply')
    voxel_path = file_path.replace('fittingPointsProj.png', 'volume_voxel.ply')
    
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Voxelization of the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    # Calculate volume
    voxel_count = len(voxel_grid.get_voxels())
    voxel_volume = voxel_size**3
    total_volume = voxel_count * voxel_volume

    # Save the original point cloud
    #o3d.io.write_point_cloud(point_cloud_path, pcd)
    
    # Save the voxelized point cloud
    #o3d.io.write_voxel_grid(voxel_path, voxel_grid)
    
    # Visualization (optional)
    #o3d.visualization.draw_geometries([voxel_grid])
    
    return total_volume

from scipy.spatial import ConvexHull

def compute_volume_with_convex_hull(points, file_path):
    """
    Compute volume of a point cloud using its convex hull.

    Parameters:
    - points: numpy array of point cloud data.
    - file_path: path to save the visualization files.

    Returns:
    - Total volume in cubic meters.
    """
    points, con_4 = hdbscan_filtering(points)
    
    # Replace the file path to save the point cloud and convex hull visualization
    point_cloud_path = file_path.replace('fittingPointsProj.png', 'volume_pc.ply')
    hull_path = file_path.replace('fittingPointsProj.png', 'volume_hull.ply')
    
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute the convex hull using open3d
    hull, _ = pcd.compute_convex_hull()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = hull.vertices
    mesh.triangles = hull.triangles
    
    # Calculate volume using scipy's ConvexHull (more direct method)
    hull_scipy = ConvexHull(points)
    total_volume = hull_scipy.volume

    # Save the original point cloud
    o3d.io.write_point_cloud(point_cloud_path, pcd)
    
    # Save the convex hull mesh
    o3d.io.write_triangle_mesh(hull_path, mesh)
    
    # Visualization (optional)
    #o3d.visualization.draw_geometries([mesh])
    
    return total_volume
