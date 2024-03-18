from RANSAC.Common import Point
from RANSAC.Common import CircleModel

from typing import List, Set, Dict, Tuple, Optional

class GradientDescentCircleFitting(object):
    """Finds the best fit circle using Gradient descent approach"""
    def __init__(self, modelhint:CircleModel, points:List[Point],learningrate:float=0.05,iterations:int=1000):
        self._modelhint=modelhint #Useful for speeding up when you have an approx circle arleady
        self._learningrate=learningrate
        self._points=points
        self._lst_distance_from_center=[] #Used as temp variable when computing MSE
        if (len(self._points) < 3):
            raise Exception("Need 3 or more points")
        self._max_iterations=iterations

    def FindBestFittingCircle(self)->CircleModel:
        seed_circle:CircleModel=None
        if (self._modelhint == None):
            seed_circle=self.GenerateSeedCircle()
        else:
            seed_circle=self._modelhint;
        for i in range(0,self._max_iterations):
            #mse=self.ComputeMse(seed_circle)
            derivative_radius,derivative_cx,derivative_cy=self.ComputeDerivativesOfMse(seed_circle)
            new_cx=-1*self._learningrate*derivative_cx + seed_circle.X
            new_cy=-1*self._learningrate*derivative_cy + seed_circle.Y
            new_radius=-1*self._learningrate*derivative_radius + seed_circle.R
            seed_circle=CircleModel(new_cx,new_cy,new_radius)
        return seed_circle
        pass

    #
    #Summation of squared difference between the following:
    #   candidate radius
    #   distance of every point from candidate center
    #
    def ComputeMse(self, candidate:CircleModel)->float:
        mse=1.0
        self._lst_distance_from_center.clear()
        for point in self._points:
            r=self.ComputeEuclideanDistance(point.X,point.Y,candidate.X,candidate.Y)            
            self._lst_distance_from_center.append(r)

        for radius in self._lst_distance_from_center:
            squared_error=(radius-candidate.R)**2
            mse+=mse+squared_error
        count_of_points=len(self._lst_distance_from_center)
        mse=mse/count_of_points * 0.5
        return mse

    #
    #Use the supplied data points to generate an approximate starting circle.
    #
    def GenerateSeedCircle(self)->CircleModel:
        all_x=list(map(lambda c: c.X, self._points))
        all_y=list(map(lambda c: c.Y, self._points))

        center_x=sum(all_x)/len(all_x)
        center_y=sum(all_y)/len(all_y)
        
        min_x=min(all_x)
        max_x=max(all_x)

        min_y=min(all_y)
        max_y=max(all_y)

        radius=abs(max_x-min_x)/2 + abs(max_y-min_y)/2
        model=CircleModel(center_x, center_y, radius)
        return model

    def ComputeEuclideanDistance(self,x1,y1,x2,y2):
        squared=(x1-x2)**2 + (y2-y1)**2
        return squared**0.5

    #
    #Compute the following derivatives of MSE
    #   derivative w.r.t to radius of candidate circle
    #   derivative w.r.t to cx of candidate circle
    #   derivative w.r.t to cy of candidate circle
    #
    def ComputeDerivativesOfMse(self,candidate:CircleModel):

        self._lst_distance_from_center.clear()
        for point in self._points:
            r=self.ComputeEuclideanDistance(point.X,point.Y,candidate.X,candidate.Y)            
            self._lst_distance_from_center.append(r)

        count_of_points=len(self._points)
        dmse_radius=0
        for radius in self._lst_distance_from_center:
            dmse_radius=dmse_radius+(radius-candidate.R)
        dmse_radius=-1*dmse_radius*(1/count_of_points)

        dmse_cx=0
        dmse_cy=0
        for idx in range(0,len(self._points)):
            pt_i=self._points[idx]
            r_i=self._lst_distance_from_center[idx]
            t1=(r_i - candidate.R) * (1/r_i)
            dmse_cx=dmse_cx + (t1 * (pt_i.X-candidate.X) * -1.0)
            dmse_cy=dmse_cy + (t1 * (pt_i.Y-candidate.Y) * -1.0)

        t2=1/count_of_points
        dmse_cx=dmse_cx * t2
        dmse_cy=dmse_cy * t2
        return dmse_radius,dmse_cx,dmse_cy
