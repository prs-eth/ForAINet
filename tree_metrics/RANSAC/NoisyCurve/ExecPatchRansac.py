from RANSAC.Algorithm import PatchRansacHelper
from RANSAC.Common import LineModel
from RANSAC.Common import Point
from RANSAC.Common import Util

import skimage
import os
import datetime
import sys
from skimage.transform import rescale, resize, downscale_local_mean
import os


def run(inputfilename,patchdimension,patchstride):
    folder_script=os.path.dirname(__file__)
    file_noisy_circle=os.path.join(folder_script,"./in/",inputfilename)
    np_image=skimage.io.imread(file_noisy_circle,as_gray=True)
    
    patcher=PatchRansacHelper(patchdimension=patchdimension,patchstride=patchstride)
    patcher.OutputImageFile=create_new_absolute_filename(inputfilename,patchdimension,patchstride)
    patcher.run(np_image)
    pass


def stride_across(image):
    pass

#Generate a dynamic file for the output rendition
def create_new_absolute_filename(inputfile,patchdimension,patchstride):
    no_extension=os.path.splitext(inputfile)[0]
    prefix=f"{no_extension}-SIZE={patchdimension}-STRIDE={patchstride}"
    folder_script=os.path.dirname(__file__)
    folder_results=os.path.join(folder_script,"./out/")
    count_of_files=len(os.listdir(folder_results))
    new_filename=("%s.%d.results.png" % (prefix,count_of_files))
    file_result=os.path.join(folder_script,"./out/",new_filename)
    return file_result

#Sine
#run("Sine-W=500.H=200.MAXD=20.SP=0.95.0.png",patchdimension=50,patchstride=25)
run("Sine-W=500.H=200.MAXD=20.SP=0.95.0.png",patchdimension=25,patchstride=12)
#Cubic
#run("Cubic-W=500.H=200.MAXD=15.SP=0.90.6.png",patchdimension=25,patchstride=12)
#run("Cubic-W=500.H=200.MAXD=15.SP=0.90.6.png",patchdimension=50,patchstride=25)
#run("Cubic-W=500.H=200.MAXD=15.SP=0.90.6.png",patchdimension=100,patchstride=50)

