import unittest
from RANSAC.Common import Util
import numpy as np
import os
import skimage
from RANSAC.Common import Point
from RANSAC.Common import Vector


class Test_test_Util(unittest.TestCase):
    #
    #Use a simple image to test the loading of points
    #
    def test_create_points_from_numpyimage(self):
        pass
        folder_script=os.path.dirname(__file__)
        filename="Util_unittest.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        height=np_image.shape[0]
        width=np_image.shape[1]

        lst_points=Util.create_points_from_numpyimage(np_image)
        np_shape=np_image.shape
        self.assertEqual(len(lst_points) , 3)

        for pt_any in lst_points:
            if pt_any.X == 0 and pt_any.Y == height-1:
                pass
            elif (pt_any.X == width-1 and pt_any.Y == height-1):
                pass
            elif (pt_any.X == width-1 and pt_any.Y == 0):
                pass
            else:
               raise Exception("Point '%s' was not expected." % (pt_any))


    
    def test_when_points_are_superimposed_over_image_array_and_saved_the_new_image_must_contain_the_new_points(self):
        folder_script=os.path.dirname(__file__)
        filename="Util_unittest.png"
        file_noisy_line=os.path.join(folder_script,"./data/",filename)
        np_image=skimage.io.imread(file_noisy_line,as_gray=True)
        file_result=os.path.join(folder_script,"../out/",filename)
        new_points=list()
        #
        #Superimpose some points
        #
        new_points.append(Point(0,0))
        new_points.append(Point(2,2))
        new_points.append(Point(3,3))
        new_points.append(Point(4,4))
        color_red=100
        color_green=255
        color_blue=90
        np_newimage=Util.superimpose_points_on_image(np_image,new_points,color_red,color_green,color_blue)
        skimage.io.imsave(file_result,np_newimage)
        #Read the image back and test the points
        np_newimage2=skimage.io.imread(file_result,as_gray=False)
        height=np_newimage.shape[0]
        for p in new_points:
            x=p.X
            y=height-p.Y-1
            self.assertEqual(np_newimage2[y][x][0],color_red)
            self.assertEqual(np_newimage2[y][x][1],color_green)
            self.assertEqual(np_newimage2[y][x][2],color_blue)
        pass
        self.assertGreater(len(new_points) , 1)
        #TODO We should verify that atleast some of the original points were re-drawn

    def test_get_extreme_colinear_points(self):
        pt1=Point(1,1)
        pt2=Point(2,2)
       
        pt3=Point(3,3)
        pt4=Point(4,4)
        pt5=Point(5,5)
        pt6=Point(6,6)

        lst_randomsequence=[pt6,pt1,pt5,pt2,pt4,pt3]
        (point1,point2)=Util.get_terminal_points_from_coliner_points(lst_randomsequence)
        results=[point1,point2]
        self.assertTrue(pt1 in results)
        self.assertTrue(pt6 in results)
        pass

    def test_generate_plottable_linear_points_between_twopoints(self):
        pt_start=Point(1,1)
        pt_end=Point(20,20)
        unit=Vector.create_vector_from_2points(pt_start,pt_end).UnitVector
        new_points=Util.generate_plottable_points_between_twopoints(pt_start,pt_end)
        distance_start_end=Point.euclidean_distance(pt_start,pt_end)
        for new_point in new_points:
            distance_from_start=Point.euclidean_distance(pt_start,new_point)
            self.assertTrue(distance_from_start < distance_start_end)
            new_unit=Vector.create_vector_from_2points(pt_start,new_point).UnitVector
            dot_product=Vector.dot_product(new_unit,unit)
            self.assertAlmostEquals(dot_product,1.0,delta=0.1)

if __name__ == '__main__':
    unittest.main()
