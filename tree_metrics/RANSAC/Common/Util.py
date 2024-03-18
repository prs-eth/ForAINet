
from .Point import Point
from .Vector import Vector
import numpy as np
from .LineModel import LineModel
from .Point import Point
from typing import Dict, List
import random


#
#Generates a list of Point classes from 2d numpy array
#The numpy array has the origin at the top left corner
#The coordinate system will be shifted to bottom-left (Cartesian)
#Only pixels with value greater than 128 will be returned
#
def create_points_from_numpyimage(arr_image:np.ndarray):
    pass
    lst=list()
    image_shape=arr_image.shape
    image_ht=image_shape[0]
    image_width=image_shape[1]
    for x in range(0,image_width):
        for y in range(0,image_ht):
            #print("x=%d y=%d" % (x,y))
            #Change coordinate system
            color=arr_image[y][x]
            #we want black pixels only
            if (color > 0.5):
                continue

            y_cartesian=image_ht - y -1
            p=Point(x,y_cartesian)
            lst.append(p)
    return lst
#
#Draws the specified collection of Point objects over the numpy array
#The coordinate system of the point will be transformed from Cartesian(bottom-left) to Image(top-left)
#
def superimpose_points_on_image(arr_image_input:np.ndarray, points,red:int,green:int,blue:int):
    width=arr_image_input.shape[1]
    height=arr_image_input.shape[0]
    arr_new=np.zeros([height,width,3])
    #We want to capture the original image
    for x in range(0,width):
        for y in range(0,height):
            color=arr_image_input[y][x]
            if (color > 0.5):
                arr_new[y][x][0]=255
                arr_new[y][x][1]=255
                arr_new[y][x][2]=255
    #superimpose the points onto the numpy array
    for p in points:
        x:int=int(round(p.X))
        y:int=int(round(height-p.Y-1))
        if (x<0 or x >= width ):
            continue
        if (y<0 or y >= height ):
            continue
        arr_new[y][x][0]=red
        arr_new[y][x][1]=green
        arr_new[y][x][2]=blue
    return arr_new

#
#The expected input image is a monochrome image
#The new points will be drawn over the old one
#
def add_points_to_monoimage(arr_image_input:np.ndarray, points,use_black:bool):
    width=arr_image_input.shape[1]
    height=arr_image_input.shape[0]
    for p in points:
        x:int=int(round(p.X))
        y:int=int(round(height-p.Y-1))
        if (x<0 or x >= width ):
            continue
        if (y<0 or y >= height ):
            continue
        arr_image_input[y][x]=0
    return arr_image_input

#
#Plots the points on the line starting from start point and leading to end point
#The points are written on to a ndarray that was originally a blank image
#The caller must take responsibility for ensuring that end coordinate values do not exceed the dimensions of the array
#
def plot_line_2darray(np_array,x_start,y_start,x_end,y_end,num_points):
    print(np_array.shape)
    print("Plotting a straight line....., ")
    xvalues = np.linspace(x_start, x_end, num_points)
    yvalues=list()
    slope=(y_end-y_start)/(x_end - x_start)
    for index in range(0,len(xvalues)):
            x=xvalues[index]
            y=slope* (x-x_start) +y_start
            print("x=%f, y=%f" % (x,y))
            yvalues.append(y)
            np_array[int(y)][int(x)][0]=0
    return np_array

#
#1)Finds the projection of the specified points on the given line
#2)Orders the projected points from in a sequence along the vector
#3)Generates new points which are uniformly distributed between the extents of the first and last point in the sequence
def generate_plottable_points_from_projection_of_points(line:LineModel,points:List[Point]):
    projected_points=LineModel.compute_projection_of_points(line,points)
    first_point,second_point=get_terminal_points_from_coliner_points(projected_points)
    plottable_points=generate_plottable_points_between_twopoints(first_point,second_point)
    return plottable_points

#def rearrange_coliner_points_sequentially(points:List[Point]):
#
#Given a list of colinear points, this function will return the 2 points which are farthest apart
#
def get_terminal_points_from_coliner_points(points:List[Point]):
    if (len(points)<2):
        raise Exception("Need 2 or more colinear points to determine sequence")
    
    if (len(points)==2):
        return (points[0],points[1])

    temp_results:List[_pairofpoints]=[]
    for i in range(0,len(points)):
        point_outer=points[i]
        for j in range(1,len(points)):
            point_inner=points[j]
            pair=_pairofpoints(point_outer,point_inner)
            temp_results.append(pair)
    pair_with_max_distance=max(temp_results,key=lambda pair: pair.distance)
    return (pair_with_max_distance.point1, pair_with_max_distance.point2)

#
#Generates a list of points between the specified start and end points
#
def generate_plottable_points_between_twopoints(point1:[Point], point2:[Point]):
    vec=Vector.create_vector_from_2points(point1,point2)
    distance_between_point1_2=Point.euclidean_distance(point1,point2)
    max_distance_between2points=1
    min_distance_between2points=max_distance_between2points*.5
    unit=vec.UnitVector
    lst_results=[]
    delta=distance_between_point1_2/10 #some random value
    last_point=point1
    last_t=0
    max_t=distance_between_point1_2
    while (True):
        new_t=last_t+delta
        if (new_t > max_t):
            break
        next_point=Point(point1.X+unit.X*new_t, point1.Y+unit.Y*new_t)
        distance=Point.euclidean_distance(last_point,next_point)
        if (distance < max_distance_between2points and distance > min_distance_between2points):
            #just right
            last_t=new_t
            lst_results.append(next_point)
            last_point=next_point
            continue
        elif (distance <= min_distance_between2points):
            #too close
            delta=delta*1.5
            continue
        elif (distance >= max_distance_between2points):
            #too far
            delta=delta*.5
            continue
        else:
            continue

    return lst_results

class _pairofpoints(object):
    def __init__(self,point1:Point, point2:Point):
        self.point1=point1
        self.point2=point2
        self._distance=-1
    
    @property
    def distance(self):
        if (self._distance != -1):
            return self._distance
        vec=Vector.create_vector_from_2points(self.point1,self.point2)
        self._distance=vec.Length
        return self._distance

    def __repr__(self):
        return f"Point1={self.point1} Point2={self.point2}"
    
