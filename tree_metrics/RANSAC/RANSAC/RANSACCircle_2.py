#
#Read the image containing the noisy 
#
import skimage
from skimage import io
import os
import datetime
import sys

from RANSAC.Common import Util
from RANSAC.Common import CircleModel
from RANSAC.Common import Point
from RANSAC.Algorithm import RansacCircleHelper
import traceback
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from RANSAC.RANSAC.MatplotUtil import plot_new_points_over_existing_points
from pathlib import Path
def run(filename, img, threshold, inlier, instance_id, sampling_fraction=0.25, matplot=True, resolution=0.005):
    print("Going to process file:%s" % (filename))
    #folder_script=os.path.dirname(__file__)
    file_noisy_circle=filename
    try:
        np_image = img

        #Iterate over all cells of the NUMPY array and convert to array of Point classes
        lst_all_points=Util.create_points_from_numpyimage(np_image)
        #print(lst_all_points)
        #
        #begin RANSAC
        #
        helper=RansacCircleHelper()
        helper.threshold_error=threshold
        helper.threshold_inlier_count=inlier
        helper.add_points(lst_all_points)
        helper.sampling_fraction=sampling_fraction
        best_model=helper.run() 
        print("RANSAC-complete") 
        if (best_model== None):
            print("ERROR! Could not find a suitable model. Try altering ransac-threshold and min inliner count")
            return -1
        #
        #Generate an output image with the model circle overlayed on top of original image
        #
        print(best_model.R*resolution)
        return_r=best_model.R*resolution
        #Load input image into array
        np_image_result=io.imread(file_noisy_circle,as_gray=True)
        new_points=CircleModel.generate_points_from_circle(best_model)
        np_superimposed=Util.superimpose_points_on_image(np_image_result,new_points,100,255,100)
        #Save new image
        #skimage.io.imsave(file_result,np_superimposed)
        #print("Results saved to file:%s" % (file_result))
        print("------------------------------------------------------------")
        if (matplot==True):
            file_result = filename.replace('fittingPointsProj', 'DBHplot')
            plot_new_points_over_existing_points(lst_all_points,new_points,"Outcome of RANSAC algorithm","Original points", "RANSAC", file_result)
        image_shape=np_image.shape
        image_ht=image_shape[0]
        real_Y=image_ht - best_model.Y -1
        return return_r, best_model.X, real_Y
    except Exception as e:
        tb = traceback.format_exc()
        print("Error:%s while doing RANSAC on the file: %s , stack=%s" % (str(e),filename,str(tb)))
        print("------------------------------------------------------------")
        #pass
        return -1, [], []



