'''
Script to execute Gradient descent fitting of a circle on a mono-chrome image
'''
import skimage
from RANSAC.Common import CircleModel
import os
import datetime
import time
import traceback
from RANSAC.Algorithm import GradientDescentCircleFitting
from RANSAC.Common import Util
from MatplotUtil import plot_new_points_over_existing_points

#
#Consumes a mono picture file and carries out RANSAC on the points and creates a new picture file
#with the same shape and new points overlayed
#
def run_image2image(filename):
    print("Going to fit circle in the file:%s" % (filename))
    folder_script=os.path.dirname(__file__)
    absolute_path=os.path.join(folder_script,"./input/",filename)
    try:
        np_image=skimage.io.imread(absolute_path,as_gray=True)
        lst_all_points=Util.create_points_from_numpyimage(np_image)
        lrate=0.3
        iterations=5000
        helper=GradientDescentCircleFitting(None,points=lst_all_points,learningrate=lrate,iterations=iterations)
        start_time = time.time()
        model:CircleModel=helper.FindBestFittingCircle()
        print("--- %s seconds for gradient descent algo ---" % (time.time() - start_time))
        #
        #Generate an output image with the model circle overlayed on top of original image
        #
        now=datetime.datetime.now()
        filename_result=("gradient-descent-%s.png" % (filename))
        file_result=os.path.join(folder_script,"./out/",filename_result)
        #Load input image into array
        np_image_result=skimage.io.imread(absolute_path,as_gray=True)
        new_points=CircleModel.generate_points_from_circle(model)
        np_superimposed=Util.superimpose_points_on_image(np_image_result,new_points,100,255,100)
        #Save new image
        skimage.io.imsave(file_result,np_superimposed)
        print("Results saved to file:%s" % (file_result))
        print("------------------------------------------------------------")

    except Exception as e:
        tb = traceback.format_exc()
        print("Error:%s while doing RANSAC on the file: %s , stack=%s" % (str(e),filename,str(tb)))
        print("------------------------------------------------------------")
        pass

    pass


def run_image2matplot(filename):
    folder_script=os.path.dirname(__file__)
    absolute_path=os.path.join(folder_script,"./input/",filename)
    try:
        np_image=skimage.io.imread(absolute_path,as_gray=True)
        lst_all_points=Util.create_points_from_numpyimage(np_image)
        plot_new_points_over_existing_points(lst_all_points,[],"Input data","Original points", "")
        lrate=0.3
        iterations=5000
        helper=GradientDescentCircleFitting(None,points=lst_all_points,learningrate=lrate,iterations=iterations)
        start_time = time.time()
        model:CircleModel=helper.FindBestFittingCircle()
        new_points=CircleModel.generate_points_from_circle(model)

        plot_new_points_over_existing_points(lst_all_points,new_points,"Gradient descent circle fitting","Original points", "Gradient descent")


    except Exception as e:
        tb = traceback.format_exc()
        print("Error:%s while doing RANSAC on the file: %s , stack=%s" % (str(e),filename,str(tb)))
        print("------------------------------------------------------------")
        pass
    pass



#run_image2image("NoisyCircle_x_32_y_28_r_25_d_0.40_sp_0.80_w_25_h_25.0.png")
#run_image2matplot("NoisyCircle_x_32_y_28_r_25_d_0.40_sp_0.80_w_25_h_25.0.png")
run_image2matplot("NoisyCircle-HandDrawn-002.png")