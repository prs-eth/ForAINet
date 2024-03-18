#
#Read the image containing the noisy 
#
import skimage
import os
import datetime

from RANSAC.Algorithm import RansacLineHelper
from RANSAC.Common import LineModel
from RANSAC.Common import Point
from RANSAC.Common import Util


def run_ransac(filename):
    folder_script=os.path.dirname(__file__)


    #Images which did not generate good results:
    #   NoisyImage_3.png
    #   NoisyLine-Gaussian-sp-0.80.111.png
    file_noisy_line=os.path.join(folder_script,"./input/",filename)
    np_image=skimage.io.imread(file_noisy_line,as_gray=True)
    #
    #Iterate over all cells of the NUMPY array and convert to array of Point classes
    #
    lst_all_points=Util.create_points_from_numpyimage(np_image)

    #
    #begin RANSAC
    #
    ransac_maxiterations=12000
    #12000
    #6000 
    #12000 worked well
    ransac_minpoints=5
    #5 worked well
    #2 gave very bad results
    #20 worked well
    ransac_threshold=5
    #25 worked well for 'NoisyLine-Gaussian-sp-0.80.104.png' 15 and 5 did not
    #Nothing worked well for 'NoisyLine-Gaussian-sp-0.80.111.png" , tried increasing to 35
    #3 for first set when points were much closer
    #5 produced too much deviation

    ransac_mininliers=10

    helper=RansacLineHelper()
    helper.max_iterations=ransac_maxiterations
    helper.min_points_for_model=ransac_minpoints
    helper.threshold_error=ransac_threshold
    helper.threshold_inlier_count=ransac_mininliers
    helper.add_points(lst_all_points)
    model=helper.run()

    #Display the model , you could render over the original picture
    print("-------------------------------------")
    print("RANSAC-complete")    
    print("Found model %s , polar=%s" % (model,model.display_polar()))
    #
    #Generate an output image with the model line 
    #
    filename_noextension=no_extension=os.path.splitext(filename)[0]
    now=datetime.datetime.now()
    filename_result=("%s-%s.result.png") % (filename_noextension,now.strftime("%Y-%m-%d-%H-%M-%S"))
    file_result=os.path.join(folder_script,"./out/",filename_result)
    #Load input image into array
    np_image_result=skimage.io.imread(file_noisy_line,as_gray=True)
    new_points=LineModel.generate_points_from_line(model,0,0,np_image_result.shape[1]-1,np_image_result.shape[0]-1)
    np_superimposed=Util.superimpose_points_on_image(np_image_result,new_points,100,255,100)
    #Save new image
    skimage.io.imsave(file_result,np_superimposed)

    #TODO YOU WILL NEED TO ASSESS HOW MANY POINTS ARE INLINERS
pass



run_ransac("NoisyLine-Gaussian-sp-0.80.103.png")
run_ransac("NoisyLine-Gaussian-sp-0.80.111.png")
run_ransac("NoisyLine-Gaussian-sp-0.80.118.png")
run_ransac("NoisyLine-Gaussian-sp-0.80.119.png")
run_ransac("NoisyLine-Gaussian-sp-0.80.121.png")
run_ransac("NoisyLine-Gaussian-sp-0.80.104.png")
