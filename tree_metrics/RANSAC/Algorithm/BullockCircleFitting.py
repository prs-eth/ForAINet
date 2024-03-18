from RANSAC.Common import Point
from RANSAC.Common import CircleModel
from typing import List, Set, Dict, Tuple, Optional
from functools import reduce

class BullockCircleFitting(object):
    """Implements Randy Bullock algorithm for circle fitting"""
    def __init__(self,points:List[Point]):
        self._points=points

        self._original_x=None
        self._original_y=None

        self._shifted_x=None
        self._shifted_y=None
        self._mean_x=0
        self._mean_y=0

    def FindBestFittingCircle(self)->CircleModel:
        self.compute_mean()
        self.shift_all_points()

        Su=sum(self._shifted_x)
        Sv=sum(self._shifted_y)
        Suu=self.compute_suu()
        Svv=self.compute_svv()
        Suv=self.compute_suv()

        Suuu=self.compute_suuu()
        Svvv=self.compute_svvv()
        Suvv=self.compute_Suvv()
        Svuu=self.compute_Svuu()
        #simulatenous equations
        #   suu*x + suv*y = c
        #   suv*x + svv*y = c
        #   uc=x (center of circle, shifted about mean)
        #   vc=y (center of circle, shifted about mean)
        #   c1 = 1/2 * (Suuu+Suvv)
        #   c2 = 1/2 * (Svvv+Svuu)
        #   a1 = suu
        #   b1 = suv
        #   a2 = suv
        #   b2 = svv

        C1 = (1/2) * (Suuu+Suvv)
        C2 = (1/2) * (Svvv+Svuu)
        Uc = (C2*Suv - C1*Svv)/(Suv*Suv - Suu*Svv)
        Vc = (C1*Suv - C2*Suu)/(Suv*Suv - Suu*Svv)
        alpha = Uc**2 + Vc**2 + (Suu+Svv)/len(self._points)

        real_x=self._mean_x+Uc
        real_y=self._mean_y+Vc
        radius = alpha**0.5

        model=CircleModel(real_x,real_y,radius)
        return model

    def compute_mean(self):
        self._original_x=list(map(lambda p:p.X,self._points))
        self._original_y=list(map(lambda p:p.Y,self._points))
        self._mean_x = sum(self._original_x)/len(self._original_x)
        self._mean_y = sum(self._original_y)/len(self._original_y)
        pass

    def shift_all_points(self):
        self._shifted_x=list(map(lambda x: x-self._mean_x,self._original_x))
        self._shifted_y=list(map(lambda y: y-self._mean_y,self._original_y))
        pass

    def compute_suu(self):
        #suu=reduce(lambda sum,x:x**2 + sum, self._shifted_x)
        suu=0
        for index in range(0,len(self._points)):
            suu=suu+(self._shifted_x[index]**2.0)
        return suu

    def compute_svv(self):
        #svv=reduce(lambda sum,y:y**2 + sum, self._shifted_y)
        svv=0
        for index in range(0,len(self._points)):
            svv=svv+(self._shifted_y[index]**2.0)
        return svv

    def compute_suv(self):
        suv=0
        for index in range(0,len(self._points)):
            suv=suv+self._shifted_x[index]*self._shifted_y[index]
        return suv

    def compute_suuu(self):
        #suuu=reduce(lambda sum,x:x**3 + sum, self._shifted_x)
        suuu=0
        for index in range(0,len(self._points)):
            suuu=suuu+self._shifted_x[index]**3
        return suuu

    def compute_svvv(self):
        #svvv=reduce(lambda sum,y:y**3 + sum, self._shifted_y)
        svvv=0
        for index in range(0,len(self._points)):
            svvv=svvv+self._shifted_y[index]**3
        return svvv

    def compute_Suvv(self):
        Suvv=0
        for index in range(0,len(self._points)):
            Suvv=Suvv+self._shifted_x[index]*self._shifted_y[index]*self._shifted_y[index]
        return Suvv

    def compute_Svuu(self):
        Svuu=0
        for index in range(0,len(self._points)):
            Svuu=Svuu+self._shifted_x[index]*self._shifted_x[index]*self._shifted_y[index]
        return Svuu

