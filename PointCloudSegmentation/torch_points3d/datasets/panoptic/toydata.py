import numpy as np
import torch
import random

from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.datasets.segmentation.toydata import toydataSphere, toydataCylinder, INV_OBJECT_LABEL
import torch_points3d.core.data_transform as cT
from torch_points3d.metrics.panoptic_tracker import PanopticTracker
from torch_points3d.metrics.panoptic_tracker_mine import MyPanopticTracker
from torch_points3d.datasets.panoptic.utils import set_extra_labels
from plyfile import PlyData, PlyElement

#-1 means unlabelled semantic classes
CLASSES_INV = {
    #0: "unclassified",
    0: "ground",
    1: "cars",
}

OBJECT_COLOR = np.asarray(
    [
        #[233, 229, 107],  # 'unclassified' .-> .yellow
        [95, 156, 196],  # 'ground' .-> . blue
        [179, 116, 81],  # 'cars'  ->  brown
        [0, 0, 0],  # unlabelled .->. black
    ]
)

VALID_CLASS_IDS = [0,1]
SemIDforInstance = np.array([1])

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

class PanoptictoydataBase:
    INSTANCE_CLASSES = CLASSES_INV.keys()
    NUM_MAX_OBJECTS = 64
    
    STUFFCLASSES = torch.tensor([i for i in VALID_CLASS_IDS if i not in SemIDforInstance])
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

        # Extract instance and box labels
        self._set_extra_labels(data)
        return data

    def _set_extra_labels(self, data):
        #return set_extra_labels(data, self.INSTANCE_CLASSES, self.NUM_MAX_OBJECTS)
        return set_extra_labels(data, self.ID2CLASS, self.NUM_MAX_OBJECTS)

    def _remap_labels(self, semantic_label):
        return semantic_label

    @property
    def stuff_classes(self):
        #return torch.tensor([0,1,5])
        return self._remap_labels(self.STUFFCLASSES)


class PanoptictoydataSphere(PanoptictoydataBase, toydataSphere):
    def process(self):
        super().process()

    def download(self):
        super().download()


class PanoptictoydataCylinder(PanoptictoydataBase, toydataCylinder):
    def process(self):
        super().process()

    def download(self):
        super().download()


class toydataFusedDataset(BaseDataset):
    """ Wrapper around toydataSphere that creates train and test datasets.

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
        dataset_cls = PanoptictoydataCylinder if sampling_format == "cylinder" else PanoptictoydataSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=30,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
            keep_instance=True,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
            keep_instance=True,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
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

    @property  # type: ignore
    @save_used_properties
    def stuff_classes(self):
        """ Returns a list of classes that are not instances
        """
        return self.train_dataset.stuff_classes

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save toydata predictions to disk using s3dis color scheme

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
        """ Allows to save toydata predictions to disk for evaluation

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
        """ Allows to save toydata instance predictions to disk using random color

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

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        #return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        return MyPanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
