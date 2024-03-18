import numpy as np
import os
import skimage
import random
import math

from RANSAC.Common import LineModel
from RANSAC.Common import Point
from RANSAC.Common import Util
from GenericCurveGenerator import GenericCurveGenerator



#
#Create blank image
#
img_back_color=255
img_width=500
img_height=200
salt_pepper_noise=0.90 #.95
max_distance_between_2_points= 20#15
#20 is a good upper limit with sp=0.95
#10 is a good lower limit, anything less then it becomes crowded

def create_new_absolute_filename(prefix):
    folder_script=os.path.dirname(__file__)
    folder_results=os.path.join(folder_script,"./out/")
    count_of_files=len(os.listdir(folder_results))
    new_filename=("%s.%d.png" % (prefix,count_of_files))
    file_result=os.path.join(folder_script,"./out/",new_filename)
    return file_result

def generate_sine():
    generator=GenericCurveGenerator(width=img_width,height=img_height)
    generator.saltpepper=0.95
    generator.curvetype="sine"
    generator.max_consecutive_distance=20
    prefix=generator.generate_filename_prefix()
    generator.output_file=create_new_absolute_filename("Sine-"+prefix)
    generator.generate_curve()
    pass

def generate_cubic():
    generator=GenericCurveGenerator(width=img_width,height=img_height)
    generator.saltpepper=0.90
    generator.curvetype="cubic"
    generator.max_consecutive_distance=15
    prefix=generator.generate_filename_prefix()
    generator.output_file=create_new_absolute_filename("Cubic-"+prefix)
    generator.generate_curve()

#generate_cubic()
generate_sine()
pass
