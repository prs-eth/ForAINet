import unittest
from RANSAC.Common import Point

class Test_test_Point(unittest.TestCase):
    def test_PointConstruction_X_Y_values_Should_Match_Constructor_Params(self):
        p1=Point(100,200)
        self.assertEqual(p1.X,100)
        self.assertEqual(p1.Y,200)

    def test_PointConstructionMultiple_ID_Must_Be_Sequential(self):
        p1=Point(100,200)
        p2=Point(100,200)
        p3=Point(100,200)
        self.assertEqual(p1.ID,p2.ID-1)
        self.assertEqual(p2.ID,p3.ID-1)

if __name__ == '__main__':
    unittest.main()
