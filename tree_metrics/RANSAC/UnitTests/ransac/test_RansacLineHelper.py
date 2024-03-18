import unittest
import math
import os
import skimage

from RANSAC.Algorithm import RansacLineHelper
from RANSAC.Common import LineModel
from RANSAC.Common import Point
from RANSAC.Common import Util

class Test_test_1(unittest.TestCase):
    def test_AddPoints(self):
        helper=RansacLineHelper()
        lst=list()
        lst.append(Point(0,0))
        lst.append(Point(1,1))
        lst.append(Point(2,2))
        lst.append(Point(3,3))
        helper.add_points(lst)
        expected_count=len(lst)
        actual_count=len(helper.points)
        self.assertEqual(expected_count,actual_count)
        pass

    def test_GetRandomPoints(self):
        helper1=RansacLineHelper()
        lst=list()
        lst.append(Point(0,0))
        lst.append(Point(1,1))
        lst.append(Point(2,2))
        lst.append(Point(3,3))
        print("displaying orignal points")
        print("------------")
        for p in lst:
            print(p)
        print("------------")
        helper1.add_points(lst)
        print("displaying points after adding to collection")
        print("------------")
        for p in helper1.points:
            print(p)
        print("------------")
        
        rnd_pts=helper1.select_random_points(2)
        self.assertEquals(len(rnd_pts), 2)
        for rnd_pt in rnd_pts:
            is_member=(rnd_pt in lst)
            print("ID of random pt=%d" % rnd_pt.ID)
            self.assertEqual(is_member,True)

    def test_create_model_vertical_through_x_equal_5(self):
        pass
        helper1=RansacLineHelper()
        lst=list()
        lst.append(Point(5,0))
        lst.append(Point(5,1))
        lst.append(Point(5,2))
        model=helper1.create_model(lst)
        expected_xintercept=5
        actual_xintercept=-model.C/model.A
        self.assertAlmostEqual(actual_xintercept,expected_xintercept)

        self.assertAlmostEqual(model.B,0)

    def test_create_model_horizontal_through_y_equal_5(self):
        pass
        helper1=RansacLineHelper()
        lst=list()
        lst.append(Point(0,5))
        lst.append(Point(1,5))
        lst.append(Point(2,5))
        model=helper1.create_model(lst)
        #y=5
        #0x+1y-5=0
        expected_slope=0.0
        actual_slope=-model.A/model.B
        actual_yintercept=-model.C/model.B
        expected_yintercept=5
        self.assertAlmostEqual(actual_slope,expected_slope)
        self.assertAlmostEqual(actual_yintercept,expected_yintercept)

    def test_create_model_45_degrees_through_origin(self):
        pass
        helper1=RansacLineHelper()
        lst=list()
        lst.append(Point(0,0))
        lst.append(Point(1,1))
        lst.append(Point(2,2))
        model=helper1.create_model(lst)
        #
        #y=x
        #-x+y+0=0
        expected_slope=1.0
        actual_slope=-model.A/model.B
        actual_yintercept=-model.C/model.B
        expected_yintercept=0
        self.assertAlmostEqual(actual_slope,expected_slope)
        self.assertAlmostEqual(actual_yintercept,expected_yintercept)

    #
    #In this test we are doing the full Ransac algorithm run on a very simple image
    #
    def test_run_with_very_simple_image(self):
        #
        #get a list of points
        #
        folder_script=os.path.dirname(__file__)
        filename_input="Line_50x30.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)
        #
        #initialize RansalHelper
        #
        helper1=RansacLineHelper()
        helper1.add_points(lst_points)
        helper1.max_iterations=1000
        #10000 did not work
        helper1.min_points_for_model=2
        helper1.threshold_error=3 #10
        helper1.threshold_inlier_count=3
        result_model=helper1.run()
        print("RANSAC-complete")    
        print("Found model %s , polar=%s" % (result_model,result_model.display_polar()))
        #
        #Superimpose the new line over the image
        #
        folder_results=os.path.join(folder_script,"../out/")
        count_of_files=len(os.listdir(folder_results))
        filename_results=("Line_50x30.%d.png" % (count_of_files) )
        file_result=os.path.join(folder_results,filename_results)
        x_lower=0
        x_upper=np_image.shape[1]-1
        y_lower=0
        y_upper=np_image.shape[0]-1
        new_points=LineModel.generate_points_from_line(result_model,x_lower,y_lower,x_upper,y_upper)
        np_superimposed=Util.superimpose_points_on_image(np_image,new_points,100,255,100)
        skimage.io.imsave(file_result,np_superimposed)
        #
        #Asserts!
        #
        x_intercept=result_model.xintercept()
        y_intercept=result_model.yintercept()
        self.assertTrue ( x_intercept > 30,"X intercept below threshold")
        self.assertTrue ( x_intercept < 50,"X intercept above threshold")
        self.assertTrue ( y_intercept > 30,"Y intercept above threshold")
        self.assertTrue ( y_intercept < 45,"Y intercept below threshold")
        self.assertTrue(len(result_model.points),5)
        for pt in result_model.points:
            distance_from_line=result_model.compute_distance(pt)
            self.assertTrue(distance_from_line <= helper1.threshold_error)
    #
    #This is an image that was taken from the sine wave picture (top left corner)
    #
    def test_run_with_50x50_image(self):
        #
        #get a list of points
        #
        folder_script=os.path.dirname(__file__)
        filename_input="Line_50x50_6pts.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)
        #
        #initialize RansalHelper
        #
        helper1=RansacLineHelper()
        helper1.add_points(lst_points)
        helper1.max_iterations=20
        helper1.min_points_for_model=2
        helper1.threshold_error=5
        helper1.threshold_inlier_count=2
        result_model=helper1.run()
        print("RANSAC-complete")    
        print("Found model %s , polar=%s" % (result_model,result_model.display_polar()))
        #
        #Superimpose the new line over the image
        #
        folder_results=os.path.join(folder_script,"../out/")
        count_of_files=len(os.listdir(folder_results))
        filename_results=("Ransac_UnitTest.Run.%d.png" % (count_of_files) )
        file_result=os.path.join(folder_results,filename_results)
        x_lower=0
        x_upper=np_image.shape[1]-1
        y_lower=0
        y_upper=np_image.shape[0]-1
        #
        #Superimpose a line over the inliers only
        #
        new_points=Util.generate_plottable_points_from_projection_of_points(result_model,result_model.points)
        np_superimposed=Util.superimpose_points_on_image(np_image,new_points,100,255,100)
        skimage.io.imsave(file_result,np_superimposed)
        pass
        #
        #No of detected inliers must be more than or equal to threshold
        #
        self.assertTrue(len(result_model.points) >= helper1.threshold_inlier_count,"Number of inliers should be >= threshold")
        #
        #There should be no-duplicates in the RANSAC inlier points
        #
        set_ids=set(map(lambda x: x.ID, result_model.points))
        list_ids=list(map(lambda x: x.ID, result_model.points))
        self.assertEqual(len(set_ids),len(list_ids),"Inliers should be unique")
        #
        #All the RANSAC linlier points must be within the threshold distance from the RANSAC line
        #
        for inlier_pt in result_model.points:
            distance=result_model.compute_distance(inlier_pt)
            self.assertTrue(distance < helper1.threshold_error,"Distance of inlier from RANSAC line must be less than threshold")

    def test_run_with_100x100_image(self):
        #
        #get a list of points
        #
        folder_script=os.path.dirname(__file__)
        filename_input="Line_100x100.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)
        #
        #initialize RansalHelper
        #
        helper1=RansacLineHelper()
        helper1.add_points(lst_points)
        helper1.max_iterations=20
        helper1.min_points_for_model=2
        helper1.threshold_error=10
        helper1.threshold_inlier_count=3
        result_model=helper1.run()
        print("RANSAC-complete")    
        print("Found model %s , polar=%s" % (result_model,result_model.display_polar()))
        #
        #Superimpose the new line over the image
        #
        folder_results=os.path.join(folder_script,"../out/")
        count_of_files=len(os.listdir(folder_results))
        filename_results=("%s.Run.%d.png" % (filename_input,count_of_files) )
        file_result=os.path.join(folder_results,filename_results)
        x_lower=0
        x_upper=np_image.shape[1]-1
        y_lower=0
        y_upper=np_image.shape[0]-1
        #
        #Superimpose a line over the inliers only
        #
        new_points=Util.generate_plottable_points_from_projection_of_points(result_model,result_model.points)
        np_superimposed=Util.superimpose_points_on_image(np_image,new_points,100,255,100)
        skimage.io.imsave(file_result,np_superimposed)
        #11 inlier points in total which give us the good model
        self.assertEqual(len(result_model.points),11)
        #
        #No of detected inliers must be more than or equal to threshold
        #
        self.assertTrue(len(result_model.points) >= helper1.threshold_inlier_count,"Number of inliers should be >= threshold")
        #
        #There should be no-duplicates in the RANSAC inlier points
        #
        set_ids=set(map(lambda x: x.ID, result_model.points))
        list_ids=list(map(lambda x: x.ID, result_model.points))
        self.assertEqual(len(set_ids),len(list_ids),"Inliers should be unique")
        #
        #All the RANSAC linlier points must be within the threshold distance from the RANSAC line
        #
        for inlier_pt in result_model.points:
            distance=result_model.compute_distance(inlier_pt)
            self.assertTrue(distance < helper1.threshold_error,"Distance of inlier from RANSAC line must be less than threshold")


if __name__ == '__main__':
    unittest.main()
