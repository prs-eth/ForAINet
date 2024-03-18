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
from RANSAC.Common import Util
import random
#
#Create blank image
#
img_back_color=255
img_width=100
img_height=50
print("This script will create a monochrome image (100X50) consisting of a random line and salt-pepper noise")
img = np.zeros([img_height,img_width,1],dtype=np.uint8)
img.fill(img_back_color)
#
#Generate Salt-Pepper noise
#
image_noisy=skimage.util.random_noise(img,mode="s&p",seed=None, clip=True,salt_vs_pepper=0.2)
#
#Generate a straight line
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

image_with_line=Util.plot_line_2darray (image_noisy,start_x,start_y,end_x,end_y,num_points)

image_result=image_with_line

#
#Save the image to disk
#
folder_script=os.path.dirname(__file__)
folder_results=os.path.join(folder_script,"./out/")
count_of_files=len(os.listdir(folder_results))

filename=("NoisyImage.%d.png" % count_of_files)
file_result=os.path.join(folder_script,"./out/",filename)
skimage.io.imsave(file_result,image_result)
print("Image saved to fileL%s" % (file_result))
#
#
#
