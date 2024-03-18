from .Point import Point
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import math

class CircleModel(object):
    """Represents a Circle model using its center point and radius"""
    def __init__(self, center_x:float,center_y:float,radius:float):
        self.X:float=center_x
        self.Y:float=center_y
        self.R:float=radius
        pass

    def __str__(self):
        display=("X=%f Y=%f R=%f" % (self.X,self.Y,self.R))
        return display

    #
    #Determine the center and radius of a circle which passes through the 3 points
    #
    @classmethod
    def GenerateModelFrom3Points(cls,p1:Point, p2:Point, p3:Point):
        x1=p1.X
        x2=p2.X
        x3=p3.X
        y1=p1.Y
        y2=p2.Y
        y3=p3.Y
        c = (x1-x2)**2 + (y1-y2)**2
        a = (x2-x3)**2 + (y2-y3)**2
        b = (x3-x1)**2 + (y3-y1)**2   
        s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c)

        px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s
        py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s 
        ar = a**0.5
        br = b**0.5
        cr = c**0.5 
        r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5

        circ=CircleModel(px,py,r)
    
        return circ


    #
    #Computes points which lie on the given circle model 
    #Returns a list of points
    #   The 'distance' governs how far apart the points would be on the circumfrence
    #
    @classmethod
    def generate_points_from_circle(cls,model,distance=1):
        angleStart=0
        angleEnd=2*math.pi
        circumfrence=2*math.pi * model.R
        num=circumfrence/distance

        angles=np.linspace(int(angleStart),int(angleEnd),int(num))
        radius=model.R
        lst_points:List[pt.Point]=list()
        for idx in range(0,len(angles)):
            theta=angles[idx]
            x=radius * math.cos(theta)  +model.X
            y=radius * math.sin(theta)  +model.Y
            pt_new=Point(x,y)
            lst_points.append(pt_new)
        return lst_points
