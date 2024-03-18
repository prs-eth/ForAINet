import laspy
import os.path
import sys
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
import csv
import random
sys.path.append("..") 
sys.path.append("../partition/cut-pursuit/build/src")
sys.path.append("../partition/ply_c")
sys.path.append("../partition")
import libcp
import libply_c
from graphs import *
from provider import *
#This file converts the las files into ply files and does some preprocessing along the way (namely it can remove points
#belonging to certain classification labels and it does a specific mapping from classification to semantic segmentation).
#It also creates a train - validation - test split from a train - test split given in data_split_metadata.csv

def print_las_info(las_file_path):
    print("las info of file: " + str(las_file_path))
    las = laspy.read(las_file_path)
    print(".las data:", las)

    for dimension in las.point_format.dimensions:
        print("Dimension name: ", dimension.name)
        print("Dimension type: ", dimension.dtype)
        print("Unique values: ", np.unique(las[dimension.name]))


def get_coord(value, scale):
    return value*scale


def las_to_ply(las_file_path, ply_file_path, merge_non_tree=False, merge_branches=False, remove_outpoints=True):
    """
    conversion from .las to .ply data type
    :param las_file_path (str): path to .las file
    :param ply_file_path (str): path where .ply file should be saved without incorporating folder name that depends on what points we have removed
    :param remove_ground (bool): whether ground points should be removed
    :param remove_lowveg (bool): whether low vegetation points should be removed
    :param remove_outpoints (bool): whether outpoints should be removed
    :return: (str) path where .ply file has actually been saved with incorporating folder name that depends on what points we have removed
    """
    las = laspy.read(las_file_path)

    scale_x, scale_y, scale_z = las.header.scale
    #we ignore the offset given in las.header.offset for each dimension, as relative position between las data files doesn't matter
    X, Y, Z = get_coord(las.X, scale_x), get_coord(las.Y, scale_y), get_coord(las.Z, scale_z)

    len = las.X.size

    #remove points from certain classification labels
    print("{} percent of the points are ground points.".format((np.count_nonzero(las.classification==2)/len)))
    print("{} percent of the points are low vegetation points.".format((np.count_nonzero(las.classification == 1) / len)))
    print("{} percent of the points are outpoints.".format((np.count_nonzero(las.classification == 3) / len)))

    ##remove points thing points without instance ids
    stuff_points = np.full(len, True)
    instance_classes = [3,4,5,6]
    for i in instance_classes:
        stuff_points = np.logical_and(stuff_points, las.classification!=i)
    stuff_instance_ids = np.unique(las.treeID[stuff_points.astype(bool)])


    foldername_addition = ""
    points_to_keep = np.full(len, True)
    if (not merge_non_tree) and (not merge_branches) and remove_outpoints: #Settting 1
        points_to_keep = np.logical_and(points_to_keep, las.classification!=3)
        foldername_addition += "_set1_5classes"
    elif merge_non_tree and (not merge_branches) and remove_outpoints: #Settting 2
        points_to_keep = np.logical_and(points_to_keep, las.classification!=3)
        foldername_addition += "_set2_4classes"
    elif merge_non_tree and merge_branches and remove_outpoints: #Settting 3
        points_to_keep = np.logical_and(points_to_keep, las.classification!=3)
        foldername_addition += "_set3_3classes"
    else:
        print("Wrong conversion settings")
        
    
    for i in instance_classes:
        points_to_keep2 = np.full(len, True)
        con_1 = las.classification==i
        points_to_keep2 = np.logical_and(points_to_keep2, con_1)
        for id in stuff_instance_ids:
            con_2 = las.treeID ==id 
            points_to_keep2 = np.logical_and(points_to_keep2, con_2)  
            points_to_keep = np.logical_and(points_to_keep, ~points_to_keep2)

    classification_new = las.classification[points_to_keep] #classification without points that should be removed
    len_new = classification_new.shape[0]
    data_struct = np.zeros(len_new, dtype=np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('return_num', 'f4'), ('num_of_return', 'f4'), ('scan_angle_rank', 'f4'), ('semantic_seg', 'f4'), ('treeID', 'f4'),
                                                    ('Sum', 'f4'), ('Omnivariance', 'f4'), ('Eigenentropy', 'f4'), ('Anisotropy', 'f4'), ('planarity', 'f4'), ('linearity', 'f4'),
                                                    ('Surface_var', 'f4'), ('scattering', 'f4'), ('verticality', 'f4'), ('Verticality2', 'f4'), ('moment1_1', 'f4'), ('moment1_2', 'f4'),
                                                    ('moment2_1', 'f4'), ('moment2_2', 'f4')]))
    data_struct['x'] = X.astype('f4')[points_to_keep]
    data_struct['y'] = Y.astype('f4')[points_to_keep]
    data_struct['z'] = Z.astype('f4')[points_to_keep]
    data_struct['intensity'] = las.intensity.astype('f4')[points_to_keep]
    data_struct['return_num'] = las.return_number.array.astype('f4')[points_to_keep]
    data_struct['num_of_return'] = las.number_of_returns.array.astype('f4')[points_to_keep]
    data_struct['scan_angle_rank'] = las.scan_angle_rank.astype('f4')[points_to_keep]
    data_struct['treeID'] = las.treeID.astype('f4')[points_to_keep]
    #data_struct['treeSP'] = las.treeSP.astype('f4')[points_to_keep]

    #adds the string foldername_addition to the data folder name to make different data folders depending on what points we have removed
    #path_to_region = ply_file_path.parents[4].joinpath(ply_file_path.parts[-5]+foldername_addition).joinpath(*ply_file_path.parts[-4:-1])
    path_to_region = ply_file_path.parents[4].joinpath(ply_file_path.parts[-5]).joinpath(*ply_file_path.parts[-4:-1])
    if not path_to_region.is_dir():
        path_to_region.mkdir(parents=True, exist_ok=True)
    ply_file_path_datanamechange = path_to_region.joinpath(ply_file_path.name)

    #mapping from classification to semantic segmentation labels (0: unclassified, 1: non-tree, 2: tree)
    sem_seg = np.full(classification_new.shape, 20.0, dtype='f4')
    if (not merge_non_tree) and (not merge_branches) and remove_outpoints: #Settting 1
        sem_seg[classification_new==0] = 0.0 #unclassified 0 -> unclassified 0
        sem_seg[classification_new==1] = 1.0 #lowveg 1-> lowveg 1
        sem_seg[classification_new==2] = 2.0 #ground 2-> ground 2
        #sem_seg[classification_new==3] = 3.0 #Outpoints have been removed
        sem_seg[classification_new==4] = 3.0 #stem points 4-> stem points 3
        sem_seg[classification_new==5] = 4.0 #live-branches 5-> live-branches 4
        sem_seg[classification_new==6] = 5.0 #branches 6-> branches 5
    elif merge_non_tree and (not merge_branches) and remove_outpoints: #Settting 2
        sem_seg[classification_new==0] = 0.0 #unclassified 0 -> unclassified 0
        sem_seg[classification_new==1] = 1.0 #lowveg 1-> non-tree 1
        sem_seg[classification_new==2] = 1.0 #ground 2 merge with lowveg -> non-tree 1
        #sem_seg[classification_new==3] = 3.0 #Outpoints have been removed
        sem_seg[classification_new==4] = 2.0 #stem points 4-> stem points 2
        sem_seg[classification_new==5] = 3.0 #live-branches 5-> live-branches 3
        sem_seg[classification_new==6] = 4.0 #branches 6-> branches 4
    elif merge_non_tree and merge_branches and remove_outpoints: #Settting 3
        sem_seg[classification_new==0] = 0.0 #unclassified 0 -> unclassified 0
        sem_seg[classification_new==1] = 1.0 #lowveg 1-> non-tree 1
        sem_seg[classification_new==2] = 1.0 #ground 2 merge with lowveg -> non-tree 1
        #sem_seg[classification_new==3] = 3.0 #Outpoints have been removed
        sem_seg[classification_new==4] = 2.0 #stem points 4-> stem points 2
        sem_seg[classification_new==5] = 3.0 #live-branches 5-> live-branches 3
        sem_seg[classification_new==6] = 3.0 #branches 6 merge with live-branches -> branches 3
    else:
        print("Wrong conversion settings - label conversion wrong")
        
    #points_except_unclassified = np.full(classification_new.shape, True)
    #points_except_unclassified = np.logical_and(points_except_unclassified, classification_new!=0)
    #u, indices = np.unique(sem_seg[points_except_unclassified], return_inverse=True)
    #data_struct['semantic_seg'][points_except_unclassified] = indices+1
    data_struct['semantic_seg']= sem_seg
    
    xyz = np.stack([data_struct[n] for n in['x', 'y', 'z']], axis=1)
    k_nn_geof = 200
    k_nn_adj = 10
    #---compute 10 nn graph-------
    graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
    #---compute geometric features-------
    geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
    
    data_struct['Sum'] = geof[:,0]
    data_struct['Omnivariance'] = geof[:,1]
    data_struct['Eigenentropy'] = geof[:,2]
    data_struct['Anisotropy'] = geof[:,3]
    data_struct['planarity'] = geof[:,4]
    data_struct['linearity'] = geof[:,5]
    data_struct['Surface_var'] = geof[:,6]
    data_struct['scattering'] = geof[:,7]
    data_struct['verticality'] = geof[:,8]
    data_struct['Verticality2'] = geof[:,9]
    data_struct['moment1_1'] = geof[:,10]
    data_struct['moment1_2'] = geof[:,11]
    data_struct['moment2_1'] = geof[:,12]
    data_struct['moment2_2'] = geof[:,13]

    del target_fea

    if (sem_seg==20.0).any():
        print("conversion not successful")

    el = PlyElement.describe(data_struct, 'vertex', comments=['Created manually from las files.'])
    PlyData([el], byte_order='<').write(ply_file_path_datanamechange)
    return ply_file_path_datanamechange

def train_val_test_split(train_test_split_path):
    """
    create a train - validation - test split from a train - test split by declaring some (original) train files as validation files
    :param train_test_split_path (str): path to data_split_metadata.csv which provides train test split
    :return:
            splitlist (list): list of all data files' relative paths, i.e. [CULS/plot_1_annotated.las, ...],
            forest_region_list (list): in which subfolder/forest region (e.g. CULS) the datafiles are,
            split_list (list): #whether the datafiels are used as train, validation or test file

    """
    csv_file = open(train_test_split_path)
    csv_reader = csv.reader(csv_file, delimiter=',')

    rel_path_list = [] #list of all data files' relative paths, i.e. [CULS/plot_1_annotated.las, ...]
    forest_region_list = [] #in which subfolder/forest region (e.g. CULS) the datafiles are
    split_list = [] #whether the datafiels are used as train, val or test file

    num_train=0
    num_test=0
    line_count = 0
    for row in csv_reader:#for each datafile
        if line_count != 0:
            if row[2]=="train":
                num_train += 1
            elif row[2]=="test":
                num_test += 1
            else:
                print("Problem: split is neither train nor test.")
        line_count += 1

    # sample randomly, but fixed (fixed because we have fixed random.seed(42) in the beginning of __main__)
    train_val_split = random.sample(range(num_train), int(0.25*num_train))
    train_val_counter = 0
    csv_file = open(train_test_split_path)
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            rel_path_list.append(row[0])
            forest_region_list.append(row[1])
            if row[2] == "test":
                split_list.append("test")
            elif row[2]=="train":
                if train_val_counter in train_val_split:
                    split_list.append("val")
                else:
                    split_list.append("train")
                train_val_counter +=1
            else:
                print("Problem: split is neither train nor test.")
        line_count += 1

    return rel_path_list, forest_region_list, split_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(42) #set seed so that validation set gets chosen randomly, but fixed from within the files annotated as
    #"train" by Stefano's train test split in data_split_metadata.csv

    #las_to_ply('/local/home/vfrawa/Documents/data/NIBIO2/plot16_annotated.las', '/local/home/vfrawa/Documents/data/NIBIO2/plot16_annotated_noground_nolowveg_nooutp.ply', True, True, True)

    #TO ADAPT: path to las data folder (data from the different regions (CULS, etc.) and data_split_metadata.csv must be in this folder)
    las_data_basepath = Path('/scratch2/OutdoorPanopticSeg_V2/data/treeinsfused/raw')
    train_test_split_path = str(las_data_basepath) + '/data_split_metadata.csv'
    rel_path_list, forest_region_list, split_list = train_val_test_split(train_test_split_path) #creates train-val-test split from train-test split
    #TO ADAPT: path where the code folder "OutdoorPanopticSeg_V2" is located
    #code_basepath = '/scratch2/OutdoorPanopticSeg_V2/data/treeinsfused/raw/las_to_ply_add_feas'
    code_basepath = '/scratch2/OutdoorPanopticSeg_V2/data_set1_5classes_200_10_allFeas/treeinsfused/raw'
    codes_data_basepath = Path(code_basepath) #this is where the ply files should be located so that the code accesses them
    #TO ADAPT: choose whether points labelled as ground, low vegetation and outpoints should be removed entirely or not
    merge_non_tree = False
    merge_branches = False
    remove_outpoints = True

    testpath_list = []
    for i in range(len(rel_path_list)): #per .las data file
        las_file_path = las_data_basepath.joinpath(rel_path_list[i])
        print(str(las_file_path))
        # print_las_info(las_file_path)
        ply_file_path = codes_data_basepath.joinpath(las_file_path.parts[-2]).joinpath(forest_region_list[i] + "_" + las_file_path.stem + "_" + split_list[i] + ".ply")
        ply_file_path_datanamechange = las_to_ply(las_file_path, ply_file_path, merge_non_tree, merge_branches, remove_outpoints)
        if split_list[i]=="test":
            testpath_list.append(str(ply_file_path_datanamechange))

    print(testpath_list) #list of paths of all files used as test files -> can be used for fold in conf/eval.yaml
