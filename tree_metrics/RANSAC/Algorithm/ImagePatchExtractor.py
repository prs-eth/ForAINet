import numpy as np
from typing import Dict, List
from RANSAC.Common import Point
from RANSAC.Common import PatchInfo,PatchResults


class ImagePatchExtractor(object):
    """slides across an image and exracts square patches"""
    def __init__(self,image,size,stride):
        self._image:np.ndarray=image
        self._stride=stride
        self._size=size

    @property
    def image(self):
        return self._image

    def extract_patches(self):
        img_width=self._image.shape[1]
        img_height=self._image.shape[0]
        _dict_patchregions=dict()
        #
        #outer while loop along Y, increment by stride
        #inner while loop along X, increment by stride
        #keep a check on x going outside of right, you will need to pad
        #How to pad? np.pad(a1,((0,0),(0,2)),'constant')
        ##

        ###

        y_boundaries:List[YBoundary]=self.calculate_y_boundaries()
        x_boundaries:List[XBoundary]=self.calculate_x_boundaries()

        arr_indices=np.zeros([len(y_boundaries),len(x_boundaries)], dtype="int")
        patch_region_index=0
        for y_index in range(0,len(y_boundaries)):
            y_boundary=y_boundaries[y_index]
            for x_index in range(0,len(x_boundaries)):
                x_boundary=x_boundaries[x_index]
                img_patch=self._image[y_boundary.y1:y_boundary.y2+1,x_boundary.x1:x_boundary.x2+1]
                pad_tuple=((0,y_boundary.overflow),(0,x_boundary.overflow))#needed for padding the leftover regions
                img_patch_padded=np.pad(img_patch,pad_tuple,'constant')
                info = PatchInfo(x_boundary.x1,y_boundary.y1,x_boundary.x2,y_boundary.y2,patch_region_index,img_patch_padded)
                _dict_patchregions[patch_region_index]=info
                arr_indices[y_index,x_index]=patch_region_index
                patch_region_index=patch_region_index+1
        r=PatchResults(_dict_patchregions,arr_indices) 
        return r
        pass

    @property
    def stride(self):
        return self._stride

    @property
    def size(self):
        return self._size

    def calculate_y_boundaries(self):
        img_height=self._image.shape[0]
        y=0
        y_boundaries:List[YBoundary]=list()
        while( y < img_height):   
            bottom=y+self._size-1
            if (bottom > img_height-1):
                #we are overshooting - re-align the filter to coincide with the bottom edge
                #padding_bottom_pixels=int((bottom-img_height))
                #y_boundaries.append(YBoundary(y,bottom,padding_bottom_pixels))
                bottom=img_height-1
                y=img_height-self.size
                y_boundaries.append(YBoundary(y,bottom,0))
                break
            else:
                y_boundaries.append(YBoundary(y,bottom,0))
                pass
            if (bottom == img_height-1):
                #The lower edge of the patch has just reached the lower edge of the image
                break
            
            y=y+self._stride
        return y_boundaries

    def calculate_x_boundaries(self):
        img_width=self._image.shape[1]
        x=0
        x_boundaries:List[XBoundary]=list()
        while( x < img_width):   
            right=x+self._size-1
            if (right > img_width-1):
                #the right edge of the patch is outside of the right edge of the image
                #padding_right_pixels=int((right-(img_width-1)))
                #x_boundaries.append(XBoundary(x,right,padding_right_pixels))
                right=img_width-1
                x=img_width-self._size
                x_boundaries.append(XBoundary(x,right,0))
                break;
            else:
                x_boundaries.append(XBoundary(x,right,0))
                pass
            if (right == img_width-1):
                #the right edge of the patch has just reached the right edge of the image
                break
            x=x+self._stride
        return x_boundaries


class YBoundary(object):
    """The upper and lower extents of the patch"""
    def __init__(self,y1,y2,overflow):
        self.y1=y1;
        self.y2=y2
        self.overflow=overflow #by how many pixels does the last patch overshoot the bottom extents

class XBoundary(object):
    """The left and right extents of the patch"""
    def __init__(self,x1,x2,overflow):
        self.x1=x1;
        self.x2=x2
        self.overflow=overflow #by how many pixels does the last patch overshoot on the right side of the border
