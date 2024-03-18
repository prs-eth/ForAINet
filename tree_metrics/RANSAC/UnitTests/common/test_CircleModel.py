import unittest
from RANSAC.Common import CircleModel
from RANSAC.Common import Point

from typing import List, Set, Dict, Tuple, Optional
import math

class Test_CircleModel(unittest.TestCase):
    def test_When_Constructed_All_Properties_Must_Be_Initialized(self):
        c1= CircleModel(100.1,200.2,300.3)
        self.assertAlmostEquals(100.1,c1.X)
        self.assertAlmostEquals(200.2,c1.Y)
        self.assertAlmostEquals(300.3,c1.R)

    def test_String_Representation_Must_Have_Center_And_Radius(self):
        c1= CircleModel(100.1,200.2,300.3)
        display=str(c1)
        self.assertTrue(display.find("X=100.1") >= 0)
        self.assertTrue(display.find("Y=200.2") >= 0)
        self.assertTrue(display.find("R=300.3") >= 0)

    #Points  (-1,0) (0,1)  (1,0)
    def test_GenerateModelFrom3pmodels_PassingThrough_1_0_And_0_1_minus1_0(self):
        p_1_0=Point(0,1)
        p_0_0=Point(1,0)
        p_minus1_0=Point(-1,0)


        c1=CircleModel.GenerateModelFrom3Points(p_0_0,p_1_0,p_minus1_0)
        #Assert on radius
        self.assertAlmostEquals(c1.R, 1.0,1)
        #Assert on center X,Y
        self.assertAlmostEquals(c1.X, 0.0,1)
        self.assertAlmostEquals(c1.Y, 0.0,1)


    #Points  (-1,1) (0,2)  (1,1)
    def test_GenerateModelFrom3pmodels_PassingThrough_1_1_And_0_2_minus1_1(self):
        p_1_0=Point(1,1)
        p_0_0=Point(0,2)
        p_minus1_0=Point(-1,1)


        c1=CircleModel.GenerateModelFrom3Points(p_0_0,p_1_0,p_minus1_0)
        #Assert on radius
        self.assertAlmostEquals(c1.R, 1.0,1)
        #Assert on center X,Y
        self.assertAlmostEquals(c1.X, 0.0,1)
        self.assertAlmostEquals(c1.Y, 1.0,1)

    def test_generate_model_from_3points_straight_line(self):
        p_0=Point(0,33)
        p_1=Point(9,30)
        p_2=Point(12,29)

        try:
            c1=CircleModel.GenerateModelFrom3Points(p_0,p_1,p_2)
            self.fail("Expected exception was not thrown")
        except:
            #Exception was expected
            pass

    #
    #A circle with center=0,0 and radius=1
    #All points generated should be 1 unit from center
    #
    def test_generate_points_from_circle_model_center_at_0_0_radius_1(self):
        center_x=0
        center_y=0
        radius=1.0
        model=CircleModel(center_x,center_y,radius)
        points=CircleModel.generate_points_from_circle(model)
        pt:Point
        for pt in points:
            distance_from_center=math.sqrt( (pt.X-center_x)**2 + (pt.Y-center_y)**2)
            self.assertAlmostEqual(distance_from_center,radius,2)

    #
    #Center with center =0,0 and radius=10
    #Generate points with 'distance' of 5
    #
    def test_generate_points_from_circle_model_center_at_0_0_radius_10(self):
        center_x=0
        center_y=0
        radius=100.0
        distance=5
        model=CircleModel(center_x,center_y,radius)
        points=CircleModel.generate_points_from_circle(model,distance=distance)
        pt:Point
        for pt in points:
            distance_from_center=math.sqrt( (pt.X-center_x)**2 + (pt.Y-center_y)**2)
            self.assertAlmostEqual(distance_from_center,radius,2)

        #compare distance
        for index in range(1,len(points)):
            point1=points[index]
            point2=points[index-1]
            actual_distance=Point.euclidean_distance(point1,point2)
            self.assertAlmostEqual(distance,actual_distance,delta=2)
            #some approximation needed due to curvature

if __name__ == '__main__':
    unittest.main()
