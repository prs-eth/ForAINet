import unittest

from RANSAC.Common import Point
from RANSAC.Common import CircleModel
from RANSAC.Algorithm import GradientDescentCircleFitting
from RANSAC.Common import Util
import os
import skimage

class Test_test_GradientDescentCircleFitting(unittest.TestCase):
    def test_Constructor_With_Learning_Rate(self):
        p1=Point(1,1)
        p2=Point(2,2)
        p3=Point(3,3)
        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
        expected_list.append(p3)

        expected_lrate=1.1
        helper=GradientDescentCircleFitting(None,points=expected_list,learningrate=expected_lrate)
        self.assertEqual(expected_lrate,helper._learningrate)
        self.assertEqual(len(expected_list),len(helper._points))
        self.assertTrue (p1 in helper._points)
        self.assertTrue(p2 in helper._points)

    def test_Constructor_With_Default_Learning_Rate(self):
        p1=Point(1,1)
        p2=Point(2,2)
        p3=Point(3,3)
        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
        expected_list.append(p3)

        expected_lrate=0.05
        helper=GradientDescentCircleFitting(None,points=expected_list)
        self.assertEqual(expected_lrate,helper._learningrate)
        self.assertEqual(len(expected_list),len(helper._points))
        self.assertTrue (p1 in helper._points)
        self.assertTrue(p2 in helper._points)

    #
    #A very basic test with the following 3 points only. 
    #   (+1,0)
    #   (-1,0)
    #   (0,1)
    #Expected:
    #   Should produce a perfect circle with center=(0,0) and radius=1.0
    #
    def test_3points_around_origin_unit_radius(self):
        p1=Point(+1,0)
        p2=Point(+0,1)
        p3=Point(-1,0)

        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
        expected_list.append(p3)

        helper=GradientDescentCircleFitting(None,expected_list)
        result:CircleModel =helper.FindBestFittingCircle()
        delta=0.01
        self.assertAlmostEquals(result.R, 1.0, delta=delta)
        self.assertAlmostEquals(result.X, 0.0, delta=delta)
        self.assertAlmostEquals(result.Y, 0.0, delta=delta)
        
    def test_5points_around_origin_unit_radius(self):
        p1=Point(+1,0)
        p2=Point(+0,1)
        p3=Point(-1,0)
        p4=Point(+0.707,+0.707)
        p5=Point(-0.707,+0.707)
        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
        expected_list.append(p3)
        expected_list.append(p4)
        expected_list.append(p5)
        helper=GradientDescentCircleFitting(None,expected_list)
        result:CircleModel =helper.FindBestFittingCircle()
        delta=0.01
        self.assertAlmostEquals(result.R, 1.0, delta=delta);
        self.assertAlmostEquals(result.X, 0.0, delta=delta);
        self.assertAlmostEquals(result.Y, 0.0, delta=delta);

    def test_5points_around_3_3_and_radius_5(self):
        #Plotted this circle using desmos.com
        #\left(5\cdot\cos\left(t\right)+3,5\cdot\sin\left(t\right)+3\right)
        p1 = Point(7,0)
        p2 = Point(8,3)
        p3 = Point(6,7)
        p4 = Point(-2,3)
        p5 = Point(0,-1)
        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
        expected_list.append(p3)
        expected_list.append(p4)
        expected_list.append(p5)
        helper=GradientDescentCircleFitting(None,expected_list)
        result:CircleModel =helper.FindBestFittingCircle()
        delta=0.01
        self.assertAlmostEquals(result.R, 5.0, delta=delta);
        self.assertAlmostEquals(result.X, 3.0, delta=delta);
        self.assertAlmostEquals(result.Y, 3.0, delta=delta);

    def test_When_GD_Is_Invoked_With_less_than_3_points_Exception_MustBe_Thrown(self):
        p1=Point(+1,0)
        p2=Point(+0,1)
        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
                
        try:
            helper=GradientDescentCircleFitting(None,expected_list)
            self.fail("Exception was expected")
        except Exception as e:
            #Ok 
            message=str(e)
            self.assertEquals(message,"Need 3 or more points");
            pass

    def test_large_circle_50X50_no_noise_1(self):
        folder_script=os.path.dirname(__file__)
        filename_input="NoisyCircle_x_-10_y_-14.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)

        helper=GradientDescentCircleFitting(None,lst_points,learningrate=0.4,iterations=5000)
        result:CircleModel =helper.FindBestFittingCircle()
        #
        #Superimpose the new line over the image
        #
        folder_results=os.path.join(folder_script,"../out/")
        count_of_files=len(os.listdir(folder_results))
        filename_results=("%s.%d.png" % (__name__,count_of_files) )
        file_result=os.path.join(folder_results,filename_results)
        new_points=CircleModel.generate_points_from_circle(result)
        np_superimposed=Util.superimpose_points_on_image(np_image,new_points,100,255,100)
        skimage.io.imsave(file_result,np_superimposed)

        delta=2
        self.assertAlmostEquals(result.R, 48.0, delta=delta);
        self.assertAlmostEquals(result.X, -10.0, delta=delta);
        self.assertAlmostEquals(result.Y, -14.0, delta=delta);
        pass

    def test_large_circle_50X50_no_noise_2(self):
        folder_script=os.path.dirname(__file__)
        filename_input="NoisyCircle_x_6_y_-30_r_118.162.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)

        helper=GradientDescentCircleFitting(None,lst_points,learningrate=0.4,iterations=5000)
        result:CircleModel =helper.FindBestFittingCircle()
        #
        #Superimpose the new line over the image
        #
        folder_results=os.path.join(folder_script,"../out/")
        count_of_files=len(os.listdir(folder_results))
        filename_results=("%s.%d.png" % (__name__,count_of_files) )
        file_result=os.path.join(folder_results,filename_results)
        new_points=CircleModel.generate_points_from_circle(result)
        np_superimposed=Util.superimpose_points_on_image(np_image,new_points,100,255,100)
        skimage.io.imsave(file_result,np_superimposed)

        delta=10
        self.assertAlmostEquals(result.R, +118.0, delta=delta);
        self.assertAlmostEquals(result.X, +06.0, delta=delta);
        self.assertAlmostEquals(result.Y, -30.0, delta=delta);
        pass

if __name__ == '__main__':
    unittest.main()
