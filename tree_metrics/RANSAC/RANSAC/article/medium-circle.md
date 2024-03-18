# Introduction

The RANSAC (Random sample and consensus) algorithm is the "gold standard" in eliminating noise. A while ago, I wrote an [article](https://medium.com/@saurabh.dasgupta1/outlier-detection-using-the-ransac-algorithm-de52670adb4a)  on how the RANSAC algorithm is implemented for finding the model of a straight line in a noisy field of points. 

The RANSAC algorithm in its original form was developed around finding straight line models when presented with noisy visual data. In this article, I will explore how to implement the RANSAC algorithm a circle model in a noisy set of points. I hope that by the end of this article you will appreciate how beautiful RANSAC is. Sometimes, I get carried away and visualize that human brain is perhaps doing some sort of RANSAC deep inside. Enough!


<img src="circle-images/HumanMind_StraightLine_Ransac.PNG" />

The example shown above exhibits how noise can impact the outcome of the least squares fitting algorithm. The human mind can easily spot the outlier, but the least squares algorithm cannot. This is where RANSAC steps in. RANSAC is a simple voting based algorithm that iteratively samples the population of points and find the subset of those lines which appear to conform.


# Problem definition
Consider the points shown below. It is not difficult to spot there is a nice circle that can be traced out of the points only if we exclude the 2 noisy points on the far right of the visual.

## Data points
<img src="circle-images/Simple.png" />

## Fitting a circle - using Gradient descent algorithm
The output after finding the best fitting circle is presented below.
<img src="circle-images/Simple_After_GradientDescent.png" />

## Fitting a circle -  using RANSAC
The output after using RANSAC to take into account the outliers. Notice that the algorithm has nicely detected the noisy points on the far right
<img src="circle-images/Simple_After_Ransac.png" />


# Understanding the RANSAC algorithm for fitting a circle

## Key definitions
<img src="circle-images/sketch.io.RANSAC-definitions.png"/>

- **Model circle**: The candidate circle which we think fits the points
- **Threshold distance**: Used for creating a doughnut like concentric circles withthe model circle in the middle
- **Outlier point**: Any point outside the doughnut region
- **Inlier point**: Any point within the doughnut region

### Mean squared error
mse = 1/N * (distance_of_point_i from center - radius)**2

## Build a shortlist of candidate circles
<< do it via Power point>>

Consider the following arrangment of points
- Select a threshold distance *threshold*
- Select a count of inliners *min inlier count*
- Select maximum number of samplings *max iterations*

pic comes here

- Select any 3 points
- find the circle passing through these 3 points
- Use *threshold* and find all points which lie in the doughnut. These are the inliers.
- If the count is less than *min inlier count* then skip this model. 
- Otherwise shortlist this model
- Use all the inlier points and find the best fitting circle. Compute mean squared error
- Repeat above steps for *max iterations* number of times

## Find the best candidate
 - Pick the circle with the max inliers
 - If there are more than 1 circles which have the same max inliers then out of the pick the one with minimum squared error

## How does the algorithm work?
<<write the algo in a formal language, wee Wikipedia >>

# Overview of the source code



# Reference
- Youtube lecture (https://www.youtube.com/watch?v=BpOKB3OzQBQ)
- Wikipedia article on RANSAC (https://en.wikipedia.org/wiki/Random_sample_consensus)
- Finding the maxima and minima (http://clas.sa.ucsb.edu/staff/lee/Max%20and%20Min's.htm)
- My original article on RANSAC for straight lines (https://medium.com/@saurabh.dasgupta1/outlier-detection-using-the-ransac-algorithm-de52670adb4a)




