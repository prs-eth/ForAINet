import unittest
from RANSAC.Common import Point
from RANSAC.Common import CircleModel
from RANSAC.Algorithm import BullockCircleFitting
from RANSAC.Common import Util
import os
import skimage

class Test_test_BullockCircleFitting(unittest.TestCase):
    #def test_A(self):
    #    self.fail("Not implemented")

    def test_constructor(self):
        p1=Point(1,1)
        p2=Point(2,2)
        p3=Point(3,3)
        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
        expected_list.append(p3)

        algo=BullockCircleFitting(expected_list)
        self.assertIsNotNone(algo)
        self.assertTrue (p1 in algo._points)
        self.assertTrue (p2 in algo._points)
        self.assertTrue (p3 in algo._points)
        self.assertEqual(len(algo._points),3)
        pass

    def test_3points_around_origin_unit_radius(self):
        p1=Point(+1,0)
        p2=Point(+0,1)
        p3=Point(-1,0)

        expected_list=list()
        expected_list.append(p1)
        expected_list.append(p2)
        expected_list.append(p3)

        helper=BullockCircleFitting(expected_list)
        result:CircleModel =helper.FindBestFittingCircle()
        delta=0.01
        self.assertAlmostEquals(result.R, 1.0, delta=delta);
        self.assertAlmostEquals(result.X, 0.0, delta=delta);
        self.assertAlmostEquals(result.Y, 0.0, delta=delta);

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
        helper=BullockCircleFitting(expected_list)
        result:CircleModel =helper.FindBestFittingCircle()
        delta=0.01
        self.assertAlmostEquals(result.R, 1.0, delta=delta);
        self.assertAlmostEquals(result.X, 0.0, delta=delta);
        self.assertAlmostEquals(result.Y, 0.0, delta=delta);

    def test_large_circle_50X50_no_noise_1(self):
        folder_script=os.path.dirname(__file__)
        filename_input="NoisyCircle_x_-10_y_-14.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename_input)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        lst_points=Util.create_points_from_numpyimage(np_image)

        helper=BullockCircleFitting(lst_points)
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

        helper=BullockCircleFitting(lst_points)
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
