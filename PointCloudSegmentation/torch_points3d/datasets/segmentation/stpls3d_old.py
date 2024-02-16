import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
# PLY reader
from torch_points3d.modules.KPConv.plyutils import read_ply, write_ply
from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

STPLS3D_NUM_CLASSES = 15

INV_OBJECT_LABEL = {
    0: "Ground",
    1: "Building",
    2: "LowVegetation",
    3: "MediumVegetation",
    4: "HighVegetation",
    5: "Vehicle",
    6: "Truck",
    7: "Aircraft",
    8: "MilitaryVehicle",
    9: "Bike",
    10: "Motorcycle",
    11: "LightPole",
    12: "StreetSgin",
    13: "Clutter",
    14: "Fence"
}


OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'Ground' .-> .yellow
        [95, 156, 196],  # 'Building' .-> . blue
        [179, 116, 81],  # 'LowVegetation'  ->  brown
        [241, 149, 131],  # 'MediumVegetation'  ->  salmon
        [81, 163, 148],  # 'HighVegetation'  ->  bluegreen
        [77, 174, 84],  # 'Vehicle'  ->  bright green
        [108, 135, 75],  # 'Truck'   ->  dark green
        [41, 49, 101],  # 'Aircraft'  ->  darkblue
        [79, 79, 76],  # 'MilitaryVehicle'  ->  dark grey
        [223, 52, 52],  # 'Bike'  ->  red
        [89, 47, 95],  # 'Motorcycle'  ->  purple
        [81, 109, 114],  # 'LightPole'   ->  grey
        [233, 233, 229],  # 'StreetSgin'  ->  light grey
        [240, 128, 128],  # Clutter -> light coral
        [0, 139, 139]  # Fence -> dark cyan
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

FILE_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
              '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
TEST_FILE_NAMES = ['5', '10', '15', '20', '25']

# FILE_NAMES = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '11', '12', '13',
#               '14', '16', '17', '18', '19', '21', '22', '23', '24']

################################### UTILS #######################################
def object_name_to_label(object_class):
    """convert from object name in NPPM3D to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["unclassified"])
    return object_label

def read_stpls3d_format(train_file, label_out=True, verbose=False, debug=False):
    """extract data from a room folder"""

    # Read object points and colors
    '''if train_file[-4::] == ".txt":
        #with open(train_file, 'r') as f:
        #    object_data = np.array([[float(x) for x in line.split()] for line in f])
        object_data = np.loadtxt(train_file, delimiter=',', skiprows=1)
        
        # Initiate containers
        cloud_points = np.empty((0, 3), dtype=np.float32)
        cloud_colors = np.empty((0, 3), dtype=np.uint8)
        cloud_classes = np.empty((0, 1), dtype=np.int32)
        cloud_instances = np.empty((0, 1), dtype=np.int32)
        cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
        cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
        cloud_classes = np.vstack((cloud_classes, object_data[:, 6].astype(np.int32).reshape(-1,1)))
        cloud_instances = np.vstack((cloud_instances, object_data[:, 7].astype(np.int32).reshape(-1,1)))
        
        # Save as ply
        write_ply(train_file[:-16]+'.ply',
                    (cloud_points, cloud_colors, cloud_classes, cloud_instances),
                    ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'label'])
        train_file = train_file[:-16]+'.ply' 
    '''
    
    raw_path = train_file
    data = read_ply(raw_path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
    rgb = np.vstack((data['red'], data['green'], data['blue'])).astype(np.uint8).T
    if not label_out:
        return (
            torch.from_numpy(xyz),
            torch.from_numpy(rgb),
            None,
            None
        )
    semantic_labels = data['class'].astype(np.int64)
    instance_labels = data['label'].astype(np.int64)
    #print(np.unique(instance_labels))
    return (
        torch.from_numpy(xyz),
        torch.from_numpy(rgb),
        torch.from_numpy(semantic_labels),
        torch.from_numpy(instance_labels),
    )


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
    el = PlyElement.describe(ply_array, "STPLS3D")
    PlyData([el], byte_order=">").write(file)
    
def to_eval_ply(pos, pre_label, gt, file):
    assert len(pre_label.shape) == 1
    assert len(gt.shape) == 1
    assert pos.shape[0] == pre_label.shape[0]
    assert pos.shape[0] == gt.shape[0]
    pos = np.asarray(pos)
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "u16"), ("gt", "u16")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["preds"] = np.asarray(pre_label)
    ply_array["gt"] = np.asarray(gt)
    PlyData.write(file)
    
def to_ins_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    max_instance = np.max(np.asarray(label)).astype(np.int32)+1
    rd_colors = np.random.randint(255, size=(max_instance,3), dtype=np.uint8)
    colors = rd_colors[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    PlyData.write(file)



################################### Used for fused STPLS3D radius sphere ###################################


class STPLS3DOriginalFused(InMemoryDataset):
    """ Original STPLS3D dataset. Each area is loaded individually and can be processed using a pre_collate transform. 
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single area by default.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 4 that denotes the area used for testing
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    #form_url = (
    #    ""
    #)
    #download_url = ""
    #zip_name = ""
    num_classes = STPLS3D_NUM_CLASSES

    def __init__(
        self,
        root,
        grid_size,
        test_area=4,
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        #assert test_area >= 1 and test_area <= 4
        if isinstance(test_area[0], int):
            assert max(test_area) <= 25
        else:
            assert len(test_area) == 1
            self.area_name = os.path.split(test_area[0])[-1].split('.')[0]
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self.grid_size = grid_size
     
        super(STPLS3DOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)
        if isinstance(test_area[0], int):

            if split == "train":
                path = self.processed_paths[0]
            elif split == "val":
                path = self.processed_paths[1]
            elif split == "test":
                path = self.processed_paths[2]
            elif split == "trainval":
                path = self.processed_paths[3]
            else:
                raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))
            self._load_data(path)

            if split == "test":
                self.raw_test_data = torch.load(self.raw_areas_paths[test_area - 1])
        else:
            #self.process_test(test_area)
            path = self.processed_paths[0]
            self._load_data(path)
            self.raw_test_data = torch.load(self.raw_areas_paths[0])

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return [osp.join(self.raw_dir, f+'.ply') for f in FILE_NAMES]

    @property
    def processed_dir(self):
        # return osp.join(self.root,'processed_'+str(self.grid_size)+'_'+str(self.test_area))
        # return osp.join(self.root,'processed_'+str(self.grid_size)+'_'+''.join(f"{a:02d}" for a in self.test_area))
        if isinstance(self.test_area[0], int):
            return osp.join(self.root,'processed_'+str(self.grid_size)+'_'+''.join(f"{a:02d}" for a in self.test_area))
        else:
            return osp.join(self.root,'processed_'+str(self.grid_size)+'_'+str(self.area_name))

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        # return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(len(FILE_NAMES))]
        if isinstance(self.test_area[0], int):
            return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(len(FILE_NAMES))]
        else:
            return [os.path.join(self.processed_dir, 'raw_area_'+self.area_name+'.pt')]

    @property
    def processed_file_names(self):
        # test_area = self.test_area
        # test_area = ''.join(f"{a:02d}" for a in self.test_area)
        # return (
        #     ["{}_{}.pt".format(s, test_area) for s in ["train", "val", "test", "trainval"]]
        #     + self.raw_areas_paths
        #     + [self.pre_processed_path]
        # )
        if isinstance(self.test_area[0], int):
            test_area = ''.join(f"{a:02d}" for a in self.test_area)
            return (
                ["{}_{}.pt".format(s, test_area) for s in ["train", "val", "test", "trainval"]]
                + self.raw_areas_paths
                + [self.pre_processed_path]
            )
        else:
            return (['processed_'+self.area_name+'.pt'])

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    #def download(self):
    #    super().download()

    def process(self):
        if not os.path.exists(self.pre_processed_path):
        
            input_ply_files =[osp.join(self.raw_dir, f+'.ply') for f in FILE_NAMES]

            # Gather data per area
            data_list = [[] for _ in range(len(input_ply_files))]
            for area_num, file_path in enumerate(input_ply_files):
            #for (area, room_name, file_path) in tq(train_files + test_files):
                xyz, rgb, semantic_labels, instance_labels = read_stpls3d_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                )

                rgb_norm = rgb.float() / 255.0


                data = Data(pos=xyz, y=semantic_labels)
                # if area_num == self.test_area-1:
                if (area_num+1) in self.test_area:
                    data.validation_set = True
                else:
                    data.validation_set = False

                if self.keep_instance:
                    data.instance_labels = instance_labels

                data.rgb = rgb_norm

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_num:")
                print(area_num)
                print("data:")  #Data(pos=[30033430, 3], validation_set=False, y=[30033430])
                print(data)
                data_list[area_num].append(data)
            print("data_list")
            print(data_list)
            raw_areas = cT.PointCloudFusion()(data_list)
            print("raw_areas")
            print(raw_areas)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])

            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    #for data in area_datas:
                    area_datas = self.pre_transform(area_datas)
            torch.save(data_list, self.pre_processed_path)
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        train_data_list = []
        val_data_list = []
        trainval_data_list = []
        for i in range(len(FILE_NAMES)):
            #if i != self.test_area - 1:
            #train_data_list[i] = []
            #val_data_list[i] = []
            for data in data_list[i]:
                validation_set = data.validation_set
                del data.validation_set
                if validation_set:
                    val_data_list.append(data)
                else:
                    train_data_list.append(data)
            trainval_data_list = val_data_list + train_data_list
        # test_data_list = data_list[self.test_area - 1]
        test_data_list = val_data_list

        #train_data_list = list(train_data_list.values())
        #val_data_list = list(val_data_list.values())
        #trainval_data_list = list(trainval_data_list.values())
        #test_data_list = data_list[self.test_area - 1]

        print("train_data_list:")
        print(train_data_list)
        print("test_data_list:")
        print(test_data_list)
        print("val_data_list:")
        print(val_data_list)
        print("trainval_data_list:")
        print(trainval_data_list)
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

    def process_test(self, test_area):

        #preprocess_dir = osp.join(self.root,'processed_'+str(self.grid_size))
        #self.processed_path = osp.join(preprocess_dir,'processed.pt')

        #if not os.path.exists(preprocess_dir):
        #    os.mkdir(preprocess_dir)
        test_data_list = []
        #self.raw_path = []
        for i, file_path in enumerate(test_area):
            area_name = os.path.split(file_path)[-1]
            if not os.path.exists(self.pre_processed_path):
                xyz, rgb, semantic_labels, instance_labels = read_stpls3d_format(
                    file_path, label_out=False, verbose=self.verbose, debug=self.debug
                )
                rgb_norm = rgb.float() / 255.0
                data = Data(pos=xyz, rgb=rgb_norm)
                if self.keep_instance:
                    data.instance_labels = instance_labels
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_name:")
                print(area_name)
                print("data:")  #Data(pos=[30033430, 3], validation_set=False, y=[30033430])
                print(data)
                test_data_list.append(data)
                # if self.pre_transform is not None:
                #     for data in test_data_list:
                #         data = self.pre_transform(data)
                #torch.save(data, pre_processed_path)
                torch.save(data, self.pre_processed_path)


            else:
                #data = torch.load(pre_processed_path)
                data = torch.load(self.pre_processed_path)
                test_data_list.append(data)

        raw_areas = cT.PointCloudFusion()(test_data_list)
        #torch.save(raw_areas, self.raw_path[0])
        torch.save(raw_areas, self.raw_areas_paths[0])

        if self.debug:
            return

        print("test_data_list:")
        print(test_data_list)
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)
        #torch.save(test_data_list, self.processed_path)
        torch.save(test_data_list, self.processed_paths[0])

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class STPLS3DSphere(STPLS3DOriginalFused):
    """ Small variation of STPLS3DOriginalFused that allows random sampling of spheres 
    within an Area during training and validation. Spheres have a radius of 8m. If sample_per_epoch is not specified, spheres
    are taken on a 0.16m grid.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 4 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=8, grid_size=0.12, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=grid_size, mode="last")
        super().__init__(root, grid_size, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def len(self):
        return len(self)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        # super().process()
        if isinstance(self.test_area[0], int):
            super().process()
        else:
            super().process_test(self.test_area)

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            #print(self._datas)
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.SphereSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class STPLS3DCylinder(STPLS3DSphere):
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
        return cylinder_sampler(area_data)

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.CylinderSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class STPLS3DFusedDataset(BaseDataset):
    """ Wrapper around STPLS3DSphere that creates train and test datasets.

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
        dataset_cls = STPLS3DCylinder if sampling_format == "cylinder" else STPLS3DSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data
        
    @property
    def test_data_spheres(self):
        return self.test_dataset[0]._test_spheres

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save STPLS3D predictions to disk using STPLS3D color scheme

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

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        #from torch_points3d.metrics.s3dis_tracker import S3DISTracker
        #return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
