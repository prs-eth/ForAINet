from typing import List, Set, Dict, Tuple, Optional
import math
import statistics as stats
import random
from RANSAC.Common import Point
from RANSAC.Common import CircleModel
from RANSAC.Algorithm import BullockCircleFitting
import threading
import sys

class TrigramOfPoints(object):
    def __init__(self,p1:Point, p2:Point, p3:Point):
        self.P1:Point=p1
        self.P2:Point=p2
        self.P3:Point=p3

        pass

class RansacCircleHelper(object):
    """Implements Ransac algorithm for Circle model"""
    def __init__(self):
        #all points in the population
        self._all_points:List[Point]=list()

        #A point will be considered an inlier of a model circle if it is 
        #within this distance from circumfrence of the model
        self.threshold_error:float=float("nan")
        
        #Minimum count of inliers needed for a model to be shortlisted
        self.threshold_inlier_count=float("nan") 

        #The learning rate used for Gradient descent circle fitting algorithm
        self.learning_rate=0.1

        self.gradient_descent_max_iterations=2000

        #The algorithm will run for these many iterations
        self.max_iterations=100

        #These many points will be selected at random to build a model
        self.min_points_for_model=20

        #Out of the total population of trigrams generated, the actual number selected is a fraction
        #specified by the following attribute
        self.sampling_fraction=0.25
        pass
    #
    #Should be called once to set the full list of data points
    #
    def add_points(self,points:List[Point]):
        self._all_points.extend(points)
        pass        

    def validate_hyperparams(self):
        if (math.isnan(self.threshold_error) == True):
            raise Exception("The property 'threshold_error' has not been initialized")

        if (math.isnan(self.threshold_inlier_count) == True):
            raise Exception("The property 'threshold_inlier_count' has not been initialized")

        if (math.isnan(self.learning_rate) == True):
            raise Exception("The property 'threshold_inlier_count' has not been initialized")
        if ((self.learning_rate <=0 ) or (self.learning_rate >= 1.0)):
            raise Exception("The attribute 'learning_rate' should be between 0 and 1")

        if (math.isnan(self.gradient_descent_max_iterations) == True):
            raise Exception("The property 'threshold_inlier_count' has not been initialized")
        
        if (math.isnan(self.sampling_fraction) == True):
            raise Exception("The property 'sampling_fraction' has not been initialized")
        if ((self.sampling_fraction <=0 ) or (self.sampling_fraction > 1.0)):
            raise Exception("The property 'sampling_fraction' should be between 0 and 1")
        

    def run(self)->CircleModel:
        self.validate_hyperparams()

        #
        #generate trigrams of points - find some temporary model to hold this model
        #
        print("Generating trigrams")
        trigrams=self.generate_trigam_from_points()
        print("Generating trigrams complete. Count=%d" % (len(trigrams)))
        #
        #for ever triagram find circle model
        #   find the circle that passes through those points
        #   Determine model goodness score
        #
        tri:TrigramOfPoints
        lst_trigram_scores=list()
        #
        all_trigram_indices=list(range(0,len(trigrams)))
        fraction=self.sampling_fraction
        random_count=int(len(all_trigram_indices)*fraction)
        if random_count>1000000:
            random_count = 1000000
        random_trigram_indices=random.sample(all_trigram_indices,random_count)
        #for trig_index in range(0,len(trigrams)):
        progress_count=0
        count_of_trigrams_with_poor_inliers=0

        #scope for improvement of performance by multithreading
        #if you use a 200X200 image, with salt peper ration of 0.85 and sample fraction of 0.2 then you can generate ample load to test multi-threading
        #
        for trig_index in random_trigram_indices:
            progress_count+=1
            tri=trigrams[trig_index]
            if (trig_index%100 ==0):
                print("PROGRESS:Processing trigram %d of %d, shortlisted=%d  poor inliers=%d" % (progress_count,len(random_trigram_indices),len(lst_trigram_scores),count_of_trigrams_with_poor_inliers))
            try:
                temp_circle=CircleModel.GenerateModelFrom3Points(tri.P1,tri.P2,tri.P3)
            except Exception as e:
                print("Could not generate Circle model. Error=%s" % (str(e)))
                continue

            inliers,goodness_score=self.get_inliers(temp_circle,[tri.P1,tri.P2,tri.P3])
            count_inliers=len(inliers)
            if (count_inliers < self.threshold_inlier_count):
                print("Skipping because of poor inlier count=%d and this is less than threshold=%f)" % (count_inliers, self.threshold_inlier_count))
                count_of_trigrams_with_poor_inliers+=1
                continue
            result=(temp_circle,inliers,tri)

            lst_trigram_scores.append(result)
        #
        #Sort trigrams with lowest error
        #
        sorted_trigram_inliercount=sorted(lst_trigram_scores, key = lambda x: len(x[1]),reverse=True)
        if (len(sorted_trigram_inliercount) ==0):
            print("Finished building shortlist of trigrams. No trigrams found. Quitting")
            return
        print("Finished building shortlist of trigrams. Count=%d, Max inlier count=%d" % (len(sorted_trigram_inliercount),len(sorted_trigram_inliercount[0][1])))
        lst_results_gdescent=list()
        jobs=[]
        for index in range(0,len(sorted_trigram_inliercount)):
            t=sorted_trigram_inliercount[index]
            model=t[0]
            inliers=t[1]
            trigram:TrigramOfPoints=t[2]
            new_points=list()
            new_points.extend(inliers)
            new_points.append(trigram.P1)
            new_points.append(trigram.P2)
            new_points.append(trigram.P3)
            new_thread=threading.Thread(target=self.find_model_using_gradient_descent2,args=(model,new_points,lst_results_gdescent))
            jobs.append(new_thread)

        for j in jobs:
            j.start();

        #Wait for all threads to finish!
        for j in jobs:
            j.join();        
        
        if (len(lst_results_gdescent) == 0):
            return None

        if (len(lst_results_gdescent) == 0):
            return None
        lst_results_gdescent_sortedby_inlier_count=sorted(lst_results_gdescent,key= lambda x: len(x[1]),reverse=True)
        max_inliers=len(lst_results_gdescent_sortedby_inlier_count[0][1])
        lst_all_results_with_highest_inlier_count= list(filter(lambda x: len(x[1])>=max_inliers, lst_results_gdescent_sortedby_inlier_count))
        lst_results_best_inlier_best_goodness=sorted(lst_all_results_with_highest_inlier_count,key= lambda x: (x[2]),reverse=False)

        best_model=lst_results_best_inlier_best_goodness[0][0]
        return best_model

    '''
    Iterates over all points and generates trigrams
    @param points:Collection of points on which to iterate
    '''
    #def GenerateTrigamFromPoints(self,points:List[Point])->List[TrigramOfPoints]:
    def generate_trigam_from_points(self,):
        #Scope for performance improvement. 
        #No need to iterate over lists 3 times. We could simply create a 3d Numpy array , 
        #the indices of the points along each of the 3 axis
        #all 3d points in this Numpy array would be our desired trigrams (except for points on all the diagonals)
        lst=list()
        
        points=self._all_points
        #print(len(points))
        if len(points)>700:
            points=random.sample(points,700)
        for i in range(0,len(points)):
            for j in range(i+1,len(points)):
                for k in range(j+1,len(points)):
                    p0=points[i]
                    p1=points[j]
                    p2=points[k]
                    trigram=TrigramOfPoints(p0,p1,p2)
                    lst.append(trigram)
                    
        random_count=int(len(lst))
        if random_count>1000000:
            random_count = 1000000
        lst=random.sample(lst,random_count)
        return lst
        pass    

    #
    #Returns all points which are within the tolerance distance from the circumfrence of the specified circle
    #Points in the exclusion list will not be considered. 
    #
    def get_inliers(self,model:CircleModel,exclude_points:List[Point])->List[Point]:
        radius=model.R
        all_points=self._all_points
        threshold=self.threshold_error
        p:Point
        shortlist_inliners=list()
        sum_goodness_measure=0
        for p in all_points:
            if (p in exclude_points):
                continue
            squared=(p.X - model.X)**2 + (p.Y - model.Y)**2 
            distance_from_center=math.sqrt(squared)
            distance_from_circumfrence=math.fabs(distance_from_center - model.R)

            if (distance_from_circumfrence > threshold):
                continue

            outlier_goodness_measure=self.compute_outlier_measure(distance_from_center,model.R)
            sum_goodness_measure+=outlier_goodness_measure

            shortlist_inliners.append(p)
        avg_goodness=1.0;
        if (len(shortlist_inliners) != 0):
            avg_goodness=sum_goodness_measure/len(shortlist_inliners)
        return (shortlist_inliners,avg_goodness)
        pass


    #
    #Gives us a relative idea of how far away the point is from the circumfrence given 
    #   the distance of the point from the center
    #   the radius of the circle
    #   Points on the circumfrence have a value of 0 increasin to 1 as we move away from the circumfrence radially
    #
    def compute_outlier_measure(self,distance,radius):
        delta=abs(distance-radius)
        mx=max(distance,radius)
        ratio=delta/mx
        return ratio

    #
    #Use the gradience descent algorithm to find the circle that fits the givens points
    #use the modelhint as a starting circle
    #
    def find_model_using_gradient_descent(self,modelhint:CircleModel, points:List[Point])->CircleModel:
        try:
            gdhelper=BullockCircleFitting.BullockCircleFitting(points)
            new_model=gdhelper.FindBestFittingCircle()
            return new_model
        except Exception as e:
            print("Error while Gradient descent: %s" % (str(e)))
            return None
        pass

    #
    #This method was written with the expectation that it would be called in the context of a new thread
    #
    def find_model_using_gradient_descent2(self,modelhint:CircleModel, points:List[Point],lst_results):
        new_model=None
        try:
            gdhelper=BullockCircleFitting.BullockCircleFitting(points)
            new_model=gdhelper.FindBestFittingCircle()
        except Exception as e:
            #print("Error while Gradient descent: %s" % (str(e)))
            pass
        if (new_model == None):
            return
        new_inliers,goodness=self.get_inliers(new_model,[])
        if (len(new_inliers) == 0):
            return
        result=(new_model,new_inliers,goodness)
        lst_results.append(result)
        return
    #
    #Returns the specified count of random selection of points from the full data set
    #
    def select_random_points(self,count:int):
        count_original=len(self._all_points)
        if (count >= count_original):
            message="The count of random points:%d canot exceed length of original list:%d" % (count,count_original)
            raise Exception(message)
        lst=random.sample(population=self._all_points,k=count)
        return lst
