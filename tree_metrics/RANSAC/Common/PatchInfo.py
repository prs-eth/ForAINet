from .Point import Point
from typing import Dict, List
import numpy as np

#
#Represents a rectangular patch in a image
#
class PatchInfo(object):
    def __init__(self,topleft_x,topleft_y,bottomright_x,bottomright_y,tag,image):
        self._topleft=Point(topleft_x,topleft_y)
        self._bottomright=Point(bottomright_x,bottomright_y)
        self._image=image #numpy array of the patch
        self._tag=tag #Some user defined tag to facilitate retrieval
        pass

    @property
    def image(self):
        return self._image

    @property
    def topleft(self):
        return self._topleft

    @property
    def bottomright(self):
        return self._bottomright

    @property
    def tag(self):
        return self._tag

    def __str__(self):
        display= ("topleft=(%d,%d)  bottomright=(%d,%d") % (self.topleft.X,self.topleft.Y,self.bottomright.X,self.bottomright.Y)
        return display

    def __repr__(self):
        display= ("topleft=(%d,%d)  bottomright=(%d,%d") % (self.topleft.X,self.topleft.Y,self.bottomright.X,self.bottomright.Y)
        return display

#
#Holds all the PatchInfo objects extracted from a single image
#
class PatchResults(object):
    def __init__(self,patches:Dict[int,PatchInfo],arrofindices:np.ndarray):
        self._patches_dict=patches
        self._2dindices=arrofindices
        pass

    #@property
    #def patches(self)->Dict[int,PatchInfo]:
    #    return self._patches_dict

    @property
    def all_patches(self):
        return list(self._patches_dict.values())

    #2d array of indices which are keys to retrieve the actual PatchInfo object at a specified x,y
    @property
    def patch_indices(self)->np.ndarray:
        return self._2dindices

    def get_patch_xy(self,xindex:int,yindex:int):
        dict_key=self._2dindices[yindex,xindex]
        return self._patches_dict[dict_key]