import unittest
from RANSAC.Common import Point
from RANSAC.Common import CircleModel
from RANSAC.Algorithm import RansacCircleHelper
import os
import skimage
from RANSAC.Common import Util

class Test_test_RansacCircleHelper(unittest.TestCase):

    def test_When_get_inliers_and_all_points_on_circumfrence_and_no_exclusion_list(self):
        #arrange
        p1=Point(+1,0)
        p2=Point(+0,1)
        p3=Point(-1,0)
        list_of_points=list()
        list_of_points.append(p1)
        list_of_points.append(p2)
        list_of_points.append(p3)

        model=CircleModel(0,0,1)
        helper=RansacCircleHelper()
        helper.threshold_error=0.2
        helper.add_points(list_of_points)

        #act
        inliers,model_score=helper.get_inliers(model,[])

        #assert
        self.assertAlmostEquals(model_score, 0.0, delta=0.1);
        self.assertEquals(len(inliers),3)
        self.assertTrue(p1 in inliers)
        self.assertTrue(p2 in inliers)
        self.assertTrue(p3 in inliers)

    def test_When_get_inliers_and_3_inliers_and_1_outlier_and_no_exclusion_list(self):
        #arrange
        p1=Point(+1.4,0.0)
        p2=Point(+0.0,1.4)
        p3=Point(-1.4,0.0)
        p_outlier=Point(-10,0)
        list_of_points=list()
        list_of_points.append(p1)
        list_of_points.append(p2)
        list_of_points.append(p3)
        list_of_points.append(p_outlier)

        model=CircleModel(0,0,1)
        helper=RansacCircleHelper()
        helper.threshold_error=0.5
        helper.add_points(list_of_points)

        #act
        inliers,model_score=helper.get_inliers(model,[])

        #assert
        expected_score=((0.4/1.4)+(0.4/1.4)+(0.4/1.4))/3.0
        self.assertAlmostEquals(model_score, expected_score, delta=0.1);
        self.assertEquals(len(inliers),3)
        self.assertTrue(p1 in inliers)
        self.assertTrue(p2 in inliers)
        self.assertTrue(p3 in inliers)
        self.assertFalse(p_outlier in inliers)

    def test_run_method_hand_drawn_circle(self):
        #
        #get a list of points
        #
        folder_script=os.path.dirname(__file__)
        filename_input="NoisyCircle-HandDrawn-001.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)
        #
        #initialize RansalHelper
        #
        helper=RansacCircleHelper()
        helper.threshold_error=20
        helper.threshold_inlier_count=5
        helper.add_points(lst_points)
        best_model=helper.run() 
        #
        #Superimpose the new line over the image
        #
        self.superimpose_circle_over_original_image(file_noisy_line,best_model)
        #
        #Assertions
        #
        delta=15
        self.assertAlmostEqual(best_model.X,89,delta=delta)
        self.assertAlmostEqual(best_model.Y,78,delta=delta)
        self.assertAlmostEqual(best_model.R,45,delta=delta)

    #
    #Superimpose the circle over the specified image file
    #
    def superimpose_circle_over_original_image(self,original_image_file,circle):
        np_image=skimage.io.imread(original_image_file,as_gray=True)

        folder_script=os.path.dirname(__file__)
        folder_results=os.path.join(folder_script,"../out/")
        count_of_files=len(os.listdir(folder_results))
        filename_results=("%s.%d.png" % (__name__,count_of_files) )
        file_result=os.path.join(folder_results,filename_results)

        new_points=CircleModel.generate_points_from_circle(circle, distance=2)
        np_superimposed=Util.superimpose_points_on_image(np_image,new_points,255,255,0)
        skimage.io.imsave(file_result,np_superimposed)

    def test_run_method_NoisyCircle2_99X50(self):
        #
        #get a list of points
        #
        folder_script=os.path.dirname(__file__)
        filename_input="NoisyCircle_99_50.png"
        file_noisy_circle=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_circle,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)
        #
        #initialize RansalHelper
        #
        helper=RansacCircleHelper()
        helper.gradient_descent_max_iterations=1000
        helper.learning_rate=0.3
        helper.threshold_error=2
        helper.threshold_inlier_count=25
        helper.max_iterations=400 #100
        helper.add_points(lst_points)
        best_model=helper.run() 
        self.superimpose_circle_over_original_image(file_noisy_circle,best_model)
        #
        #Assertions
        #
        delta=10
        self.assertAlmostEqual(best_model.X,61,delta=delta)
        self.assertAlmostEqual(best_model.Y,129,delta=delta)
        self.assertAlmostEqual(best_model.R,111,delta=delta)


    def test_run_method_NoisyCircle1_50X50(self):
        #
        #get a list of points
        #
        folder_script=os.path.dirname(__file__)
        filename_input="NoisyCircle_x_116_y_-15_r_133_d_0.500000_sp_0.5.177.png"
        file_noisy_circle=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_circle,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)
        #
        #initialize RansalHelper
        #
        helper=RansacCircleHelper()
        helper.gradient_descent_max_iterations=1000
        helper.learning_rate=0.3
        helper.threshold_error=2
        helper.threshold_inlier_count=15
        helper.max_iterations=400 #100
        helper.add_points(lst_points)
        best_model=helper.run() 
        self.superimpose_circle_over_original_image(file_noisy_circle,best_model)
        #
        #Assertions
        #
        delta=10
        self.assertAlmostEqual(best_model.X,115,delta=delta)
        self.assertAlmostEqual(best_model.Y,-64,delta=delta)
        self.assertAlmostEqual(best_model.R,131,delta=delta)


    def test_run_method_LargeCircle1_50X50_NoNoise(self):
        #
        #get a list of points
        #
        folder_script=os.path.dirname(__file__)
        filename_input="NoisyCircle_x_-10_y_-14.png"
        file_noisy_circle=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_circle,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)
        #
        #initialize RansalHelper
        #
        helper=RansacCircleHelper()
        helper.gradient_descent_max_iterations=1000
        helper.learning_rate=0.3
        helper.threshold_error=5
        helper.threshold_inlier_count=15
        #helper.max_iterations=400 #100
        helper.add_points(lst_points)
        best_model=helper.run() 
        self.superimpose_circle_over_original_image(file_noisy_circle,best_model)
        #
        #Assertions
        #
        delta=15
        self.assertAlmostEqual(best_model.X,-2,delta=delta)
        self.assertAlmostEqual(best_model.Y,-5,delta=delta)
        self.assertAlmostEqual(best_model.R,37,delta=delta)
        pass


if __name__ == '__main__':
    unittest.main()

    #you were going to test "NoisyCircle_x_116_y_-15_r_133_d_0.500000_sp_0.5.177"
