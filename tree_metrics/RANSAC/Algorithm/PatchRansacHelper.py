import numpy as np
from typing import Dict, List
from RANSAC.Common import LineModel
from RANSAC.Algorithm import RansacLineHelper
from RANSAC.Common import Util
from RANSAC.Common import Point
from RANSAC.Common import PatchInfo,PatchResults
from .ImagePatchExtractor import ImagePatchExtractor
import skimage

class PatchRansacHelper(object):
    """description of class"""
    def __init__(self,patchdimension,patchstride):
        self._image:np.ndarray=None

        if (patchdimension<=0):
            raise Exception("The dimension of the patch should be a positive number")
        else:
            self._cropdimension=patchdimension #the size of the patch square

        if (patchstride<=0):
            raise Exception("The stride should be a positive number")
        else:
            self._stride=patchstride #the number of pixels to advance the patch window
    
        self._dict_patchregions:Dict=dict()        
        self.OutputImageFile=None #The absolute path to the file where the overlaid image will be saved
        pass
    
    @property
    def patchdimension(self):
        return self._cropdimension

    @property
    def patchstride(self):
        return self._stride

    @property
    def image(self):
        return self._image

    def run(self,image):
        print(f"Going to run RANSAC on a patch square with dimension:{self.patchdimension}  and stride:{self.patchstride}")
        self._image=image
        xtractor=ImagePatchExtractor(self.image,self._cropdimension,self._stride)
        patch_results:PatchResults=xtractor.extract_patches();
        patchcount_x=patch_results.patch_indices.shape[1]
        patchcount_y=patch_results.patch_indices.shape[0]
        good_patches:List[_PatchAnalysis]=list()
        for x in range(0,patchcount_x):
            for y in range(0,patchcount_y):
                patchinfo:PatchInfo=patch_results.get_patch_xy(x,y)
                img_patchregion=patchinfo.image
                lst_all_points=Util.create_points_from_numpyimage(img_patchregion)
                line=self.find_line_using_ransac(lst_all_points,img_patchregion)
                if (line  == None):
                    continue
                print("Got a line X=%d Y=%d , line=%s" % (x,y, str(line)))
                #add to a collection of patch regions+ransacline
                patch_analysis_result=_PatchAnalysis(patchinfo,lst_all_points,line)
                good_patches.append(patch_analysis_result)
        pass
        print("Total interesting patches with lines found = %d" % (len(good_patches)))
        #you have all the patches that yielded some lines, superimpose on the original
        for good_patch in good_patches:
            print(str(good_patch))
        self.__superimpose(good_patches)

    def __superimpose(self,patches):
        lst_allpoints_from_all_patches=list()
        image_height=self.image.shape[0]
        for patch in patches:
            try:
                ransac_points:List[Point]=patch.ransacline.points
                plottable_points=Util.generate_plottable_points_from_projection_of_points(patch.ransacline,patch.ransacline.points)
                for plottable_point in plottable_points:
                    new_x=patch.patchinfo.topleft.X + plottable_point.X
                    #new_y=patch.patchinfo.topleft.Y + plottable_point.Y
                    #new_y=(image_height-patch.patchinfo.topleft.Y) + plottable_point.Y
                    new_y=(image_height-patch.patchinfo.topleft.Y-self._cropdimension) + plottable_point.Y
                    translated_point=Point(new_x,new_y)
                    lst_allpoints_from_all_patches.append(translated_point)
            except Exception  as e:
                print("Exception while superimposing %s" % (patch.patchinfo))
        pass
        np_superimposed_patches=Util.superimpose_points_on_image(self._image,lst_allpoints_from_all_patches,100,255,100)
        skimage.io.imsave(self.OutputImageFile,np_superimposed_patches)
        pass


    #
    #Look for a line using RANSAC within each patch region
    #
    def find_line_using_ransac(self,points_in_patch:List[Point],np_patchregion:np.ndarray):
        
        count_of_points_in_patch=len(points_in_patch)
        

        #create thresholds for RANSAC
        ransac_threshold_distance=(self._cropdimension)*1/10 #should be adaptive
        ransac_mininliers=0.5 * count_of_points_in_patch #automatically try out various inliers

        helper=RansacLineHelper()
        helper.max_iterations=(count_of_points_in_patch)*(count_of_points_in_patch-1)/2 * 5 #mpy by 5
        helper.min_points_for_model=2
        helper.threshold_error=ransac_threshold_distance
        helper.threshold_inlier_count=ransac_mininliers
        helper.add_points(points_in_patch)
        model=helper.run()
        return model 
        pass

#Represents a dto class which holds the original PatchInfo,points in this patch and the RANSAC line
class _PatchAnalysis(object):
    def __init__(self,patchinfo:PatchInfo,points_in_patch:List[Point],line:LineModel):
        self.patchinfo=patchinfo
        self.points=points_in_patch
        self.ransacline:LineModel=line
        pass

    def __str__(self):
        display= ("Line=%s Points in line=%d  Total points in patch=%d   PatchInfo=%s") % (self.ransacline,len(self.ransacline.points), len(self.points),self.patchinfo)
        return display
