import statistics 
import random
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from RANSAC.Common import Point
from RANSAC.Common import LineModel
from typing import Dict, List

class RansacLineHelper(object):
    """Encapsulates RANSAC logic"""
    def __init__ (self):
        pass
        self.__complete_list_of_points:list=list()
        self.__max_iterations:float=0
        self.min_points_for_model:float=0
        # 'threshold_error' is the threshold distance from a line for a point to be classified as an inlier
        self.threshold_error:float=0 #max distance of a point from the line to be considered as inlier
        self.threshold_inlier_count:int=0 #minimum number of inliers for a model to be considered as good
    
    #
    #Should be called once to set the full list of data points
    #
    def add_points(self,points:list):
        self.points.extend(points)
        pass
    #
    #Get the collection of points
    #
    @property
    def points(self):
        return self.__complete_list_of_points

    @property
    def max_iterations(self):
        return self.__max_iterations

    @max_iterations.setter
    def max_iterations(self,value):
        self.__max_iterations=value

    #
    #Executes the algorithm
    #
    def run(self) -> LineModel:
        if (len(self.points) <=2 ):
            return None
        temporary_models=self.__create_all_temp_models()
        if (len(temporary_models) == 0):
            return None
        
        temporary_models_high_inliers=list(filter(lambda  m: len(m.points) >= self.threshold_inlier_count,temporary_models))
        if (len(temporary_models_high_inliers) == 0):
            return None

        expanded_models=self.__create_expanded_models(temporary_models_high_inliers)
        if (len(expanded_models) == 0):
            return None
        best_model=self.__get_model_with_maxinliers_and_lowest_error(expanded_models)
        return best_model.linemodel

    #
    #Get all modesl which have inliers above specified threshold
    #
    def __create_all_temp_models(self)->List[LineModel]:
        lst_results:List[LineModel]=list()
        iter=0
        while (iter < self.max_iterations):
            print("-------------------------------------")
            iter+=1
            random_points=self.select_random_points(self.min_points_for_model)
            print("Found %d random points" % len(random_points))
            temp_model=self.create_model(random_points)
            print("Built model %s using %d random points" % (temp_model.display_polar(),len(random_points)))
            inliers=self.get_inliers_from_model(temp_model,[])
            temp_model.points.extend(inliers)
            lst_results.append(temp_model)

        return lst_results
    
    #
    #Given the models found in the first pass, 
    #Use all the inliers and find a better line model using least squares which should be better than the original model
    #
    def __create_expanded_models(self,models:List[LineModel]):
        lst_results:List[ExpandedModel]=list()
        for model in models:
            better_model=self.create_model(model.points)
            inliers_better_model=self.get_inliers_from_model(better_model,[])
            if (len(inliers_better_model) == 0):
                continue
            model_error=self.compute_mean_squared_distance(better_model,inliers_better_model)
            better_model.points.extend(inliers_better_model)
            expanded_model=ExpandedModel(better_model,model_error)
            lst_results.append(expanded_model)
        return lst_results

    def __get_model_with_maxinliers_and_lowest_error(self, models):
        model_with_maxinliers=max(models, key=lambda x: x.count_of_inliers)
        max_inlier_count=model_with_maxinliers.count_of_inliers
        lst_all_results_with_highest_inlier_count=list(filter(lambda x: x.count_of_inliers==max_inlier_count, models))
        lst_sorted_results_with_highest_inlier_count=sorted(lst_all_results_with_highest_inlier_count,key=lambda  x: x.error)
        final_model=lst_sorted_results_with_highest_inlier_count[0]
        return final_model

    #
    #The mean of the squared distances. This will penalize points which are farther away
    #
    def compute_mean_squared_distance(self,model:LineModel,points:list) -> float:
        lst_distances=list()
        for p in points:
            distance=model.compute_distance(p)
            lst_distances.append(distance**2)
        sum_of_squared_distances=sum(lst_distances)
        sqroot=sum_of_squared_distances**0.5
        mean=sqroot/len(lst_distances)
        return mean

    def compute_average_distance(self,model:LineModel,points:list) -> float:
        lst_distances=list()
        for p in points:
            distance=model.compute_distance(p)
            lst_distances.append(distance)
        mean=statistics.mean(lst_distances)
        return mean
    #
    #Get all points from master list (not used for model building) within error threshold
    #
    def get_inliers_from_model(self,model:LineModel,points_old_inliers:list) -> list:
        lst_inliers=list()
        for p in self.points:
            if ((p in points_old_inliers) == True):
                continue
            distance_from_model:float=model.compute_distance(p)
            if (distance_from_model > self.threshold_error):
                continue
            lst_inliers.append(p)
        return lst_inliers
    #
    #Returns the specified count of random selection of points from the full data set
    #
    def select_random_points(self,count:int):
        #Temporary implementation only
        count_original=len(self.points)
        if (count >= count_original):
            message="The count of random points:%d canot exceed length of original list:%d" % (count,count_original)
            raise Exception(message)
        lst=random.choices(population=self.points,k=count)
        return lst
    #
    #Find the best line which fits the specified points
    #Use the least squares best fit
    #https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
    #
    def create_model(self,points:list)->LineModel:

        mean_x=0
        mean_y=0
        for p in points:
            mean_x+=p.X
            mean_y+=p.Y
        mean_x=mean_x/len(points)
        mean_y=mean_y/len(points)

        slope_numerator=0
        slope_denominator=0
        slope=0
        #use the formula for least squares
        for p in points:
            slope_numerator+=(p.X-mean_x)*(p.Y-mean_y)
            slope_denominator+=(p.X-mean_x)*(p.X-mean_x)

        if (math.fabs(slope_denominator) < 0.001):
            #perpendicular line
            x_intercept=mean_x
            #equation   (1)x + (0)y + (-xintercept) + 1
            vertical_line_a=1
            vertical_line_b=0
            vertical_line_c=-x_intercept
            model=LineModel(vertical_line_a,vertical_line_b,vertical_line_c)
            return model
        
        slope=slope_numerator/slope_denominator
        y_intercept=mean_y - (slope * mean_x)

        line_a=slope
        line_b=-1
        line_c=y_intercept
        #  standard form of line equation
        #  ------------------------------
        #   y=mx+c
        #   mx  -   y   +   c=0
        #   ax  +   by  +   c=0
        #   slope= -a/b
        #   yint= -c/b
        #        
        model=LineModel(line_a,line_b,line_c)
        return model
#
#Holds a line equation, inlier points and mean squared error of the inliers
#
class ExpandedModel(object): 
    def __init__(self,line:LineModel,error:float):
        self.linemodel=line
        self.error=error
        self.count_of_inliers=len(self.linemodel.points)
            
        pass
    def __repr__(self):
        return f"line={self.linemodel}  error={self.error}   inlier_count={self.count_of_inliers}"
