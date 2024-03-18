import unittest
from RANSAC.Common import Point
from RANSAC.Common import Vector

class Test_test_Vector(unittest.TestCase):
    def test_create_vector(self):
        p1=Point(1,1)
        p2=Point(2,2)
        v=Vector.create_vector_from_2points(p1,p2)
        self.assertEquals(v.X,1)
        self.assertEquals(v.Y,1)

    def test_vector_construction(self):
        v=Vector(100,200)
        self.assertEquals(v.X,100)
        self.assertEquals(v.Y,200)

    def test_vector_length(self):
        v=Vector(3,4)
        self.assertEquals(v.Length,5)

if __name__ == '__main__':
    unittest.main()
