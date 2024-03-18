#
#Use this script to generate a mono chrome picture with 1 line and noise
#This will generate a random straight line 
#  *)left edge to right edge
#  *)or top edge to bottom edge
#  *)scatter the picture with salt-pepper noise
#

import numpy as np
import os
import skimage
import random

from RANSAC.Common import LineModel
from RANSAC.Common import Point
from RANSAC.Common import Util
from RANSAC.Common import GenerateGaussianNoiseAtPoint as gnp

#
#Create blank image
#
img_back_color=255
img_width=100
img_height=50
salt_pepper_noise=.8 #0.2 Temporary
print("This script will create a monochrome image (100X50) consisting of a random line and salt-pepper noise")
img = np.zeros([img_height,img_width,1],dtype=np.uint8)
img.fill(img_back_color)
#
#Generate Salt-Pepper noise
#
image_noisy=skimage.util.random_noise(img,mode="s&p",seed=None, clip=True,salt_vs_pepper=salt_pepper_noise)
#
#Generate noisy line
#
num_points=30
if (random.randint(0,1) == 0):
    #generate a line starting from a random point at the bottom edge and going up to a random point on the top edge
    start_x=random.random() * (img_width-1)
    start_y=0
    end_x=random.random() * (img_width-1)
    end_y=img_height-1
else:
    #generate a line starting from a random point on the left edge and going up to a random point on the right edge
    start_x=0
    start_y=random.random() * (img_height-1)
    end_x= img_width-1
    end_y=random.random() * (img_height-1)
#
#Generate gaussian points
#
gap=5
num_gaussian_points=5 #3
line_model=LineModel.create_line_from_2points(start_x,start_y,end_x,end_y)
pts_st_line_regular_intervals=LineModel.generate_points_at_regular_intervals_stline(start_x,start_y,end_x,end_y,gap)
print("Calculate %d points at spacing of %.f" % (len(pts_st_line_regular_intervals),gap))
pts_new=list();
for pt_regular in pts_st_line_regular_intervals:
    arr_cluster=gnp.GenerateClusterOfRandomPointsAroundXY(pt_regular.X,pt_regular.Y,gap,num_gaussian_points)
    cluster_shape=arr_cluster.shape
    for idx in range(0,cluster_shape[0]):
        x_cluster=arr_cluster[idx][0]; 
        y_cluster=arr_cluster[idx][1];
        pt_noisy=Point(x_cluster,y_cluster)
        pts_new.append(pt_noisy)
print("Calculated %d noisy points" % (len(pts_new)))
image_result=Util.superimpose_points_on_image(image_noisy,pts_new, 0,0,0)
######################
#
#Save the image to disk
#
folder_script=os.path.dirname(__file__)
folder_results=os.path.join(folder_script,"./out/")
count_of_files=len(os.listdir(folder_results))

filename=("NoisyLine-Gaussian-sp-%.2f.%d.png" % (round(salt_pepper_noise,2),count_of_files))
file_result=os.path.join(folder_script,"./out/",filename)
skimage.io.imsave(file_result,image_result)
print("Image saved to fileL%s" % (file_result))


#arr_noisy=gnp.GenerateClusterOfRandomPointsAroundXY(0,0,5,10)

#arr_noisy=GenerateGaussianNoiseAtPoint.GenerateClusterOfRandomPointsAroundXY(0,0,5,10)
pass


#YOU WERE HERE , YOU WERE GOING TO CREATE A NOISY LINE USING THIS
