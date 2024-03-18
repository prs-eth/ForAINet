from .Point import Point
import math
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
from scipy import linalg


class LineModel:
    """Describes the equation of a straight line in ax+by+c format"""
    SMALL=0.001
    def __init__ (self, a:float,b:float,c:float):
        self.A=a
        self.B=b
        self.C=c
        self.__points:List[Point]=list() #Optional the points that were used to make this line
        pass

    def __str__(self):
        display= ("A=%f B=%f C=%f") % (self.A,self.B,self.C)
        return display

    def __repr__(self):
        display= ("A=%f B=%f C=%f") % (self.A,self.B,self.C)
        return display

    @property
    def points(self)->List[Point]:
        return self.__points

    #
    #Compute distance of the point from the model line
    #
    def compute_distance(self,point:Point):
        numerator=self.A*point.X + self.B*point.Y + self.C
        denominator=math.sqrt(self.A*self.A + self.B*self.B)
        distance=math.fabs( numerator/denominator)
        return distance

    def get_slope(self):
        if (math.fabs( self.B ) < 0.001):
            raise Exception("Infinit slope")
        else:
            slope=-self.A/self.B
            return slope
    #
    #Returns inclination of the line with positive X axis
    #
    def get_inclination(self):
        if (math.fabs( self.B ) < 0.001):
            return math.pi/2;
        else:
            slope=-self.A/self.B
            angle=math.atan(slope)
            return angle
    #
    #Gets the Y intercept of the line, if infinite then math.inf
    #
    def yintercept(self):
        intercept=math.inf
        if (math.fabs(self.B) > LineModel.SMALL):
            intercept=-self.C/self.B;
        return intercept
    #
    #Gets the X intercept of the line, if infinite then math.inf
    #
    def xintercept(self):
        intercept=math.inf
        if (math.fabs(self.A) > LineModel.SMALL):
            intercept=-self.C/self.A
        return intercept

    #
    #Create a friendly display which shows all attributes of the Line
    #
    def display_polar(self):
        origin=Point(0,0)
        distance_origin=self.compute_distance(origin)
        deg_per_radian=90*2/math.pi;
        angle=self.get_inclination() * deg_per_radian; 

        #   slope= -a/b
        #   yint= -c/b
        #   xint= -c/a
        y_intercept=self.yintercept()
        x_intercept=self.xintercept()

        display=("POLAR radius=%f  angle=%f     y_int=%s    x_int=%s" % (distance_origin,angle,y_intercept,x_intercept))
        return display

    #
    #This function will generate points using the specified line model within the bounds of x1,y1 and x2,y2
    #Returns an array of Point class instances
    #We increment X or Y by delta_increment= pixel and generate new points
    #
    @classmethod
    def generate_points_from_line(cls,model,x1:float,y1:float,x2:float,y2:float)->List[Point]:
        lst_points:List[Point]=list()
        delta_increment=0.01
        if (math.fabs(model.B) < LineModel.SMALL):
            #
            #Perp line
            #
            min_y=min(y1,y2)
            max_y=max(y1,y2)
            start_y=min_y
            while (start_y <= max_y):
                #ax+by+c=0
                #x=(-c-by)/a
                x=(-model.C - 0)/model.A
                pt_new=Point(x,start_y)
                lst_points.append(pt_new)
                start_y+=delta_increment
        else:
            #
            #All other points
            #
            min_x=min(x1,x2)
            max_x=max(x1,x2)
            start_x=min_x
            while (start_x <= max_x):
                #ax+by+c=0
                #y= (-c -ax)/b
                y=(-model.C - model.A*start_x)/model.B
                pt_new=Point(start_x,y)
                lst_points.append(pt_new)
                start_x+=delta_increment
        return lst_points
 
    #
    #Creates a line equation from start and end points
    #Returns a LineModel instance
    #
    @classmethod
    def create_line_from_2points(cls,x_start,y_start,x_end,y_end):
        is_infinite_slope=abs(x_start-x_end) < cls.SMALL
        if (is_infinite_slope):
            #nearly vertical, take the average of x
            x_intercept=(x_start+x_end)/2
            line_a=1
            line_c=-1 * line_a * x_intercept
            line_b=0
            model=cls(line_a,line_b,line_c)
            return model    
        else:
            slope=(y_end-y_start)/(x_end-x_start)
            y_intercept= y_end - slope * x_end
            line_a=slope
            line_b=-1
            line_c=y_intercept
            model=cls(line_a,line_b,line_c)
            return model    
        pass
    #
    #Returns a list of Points which are generated on a straight line connecting start and end
    #The points are spaced at intervals of 'gap' distance
    #
    @classmethod
    def generate_points_at_regular_intervals_stline(cls,start_x:float,start_y:float,end_x:float,end_y:float,gap:float):
        model=cls.create_line_from_2points(start_x,start_y,end_x,end_y)
        pts_result=list()
        pts_crowded=cls.generate_points_from_line(model,start_x,start_y,end_x,end_y)
    
        point_last=pts_crowded[0]
        for index in range(1,len(pts_crowded)):
            point_current=pts_crowded[index]
            distance= math.sqrt( (point_current.X-point_last.X)**2 + (point_current.Y-point_last.Y)**2 )
            if (distance > gap):
                pts_result.append(point_current)
                point_last=point_current
                continue
            else:
                pass
        return pts_result

    #
    #Determines the projection of the specified points on the given line
    #Returns a list of projected Point objects
    #
    @classmethod
    def compute_projection_of_points(cls,line,points:List[Point]):
        lst_projections=list()
        for pt_input in points:
            #determine equation in ax+by+c=0
            a_perp=line.B
            b_perp=-line.A
            c_perp=-a_perp*pt_input.X - b_perp*pt_input.Y
            line_perp=LineModel(a_perp,b_perp,c_perp)
            #The intersection of line and line_perp is the projection of pt_input on the given line
            arr_lhs=np.array([[line.A,line.B],[line_perp.A,line_perp.B]])
            arr_rhs=np.array([-line.C,-line_perp.C])
            intersection=np.linalg.solve(arr_lhs,arr_rhs)
            lst_projections.append(Point(intersection[0],intersection[1]))
        return lst_projections
        pass
