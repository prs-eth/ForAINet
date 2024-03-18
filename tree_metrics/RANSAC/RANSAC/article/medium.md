# Introduction
In this article we will explore the **Random Sample Consensus** algorithm - more popularly known by the acronym RANSAC. This is an iterative and a non-deterministic algorithm that helps in eliminating outliers from a dataset. This algorithm is commonly used  to solve computer vision challenges. In this article I have presented the motivation for the RANSAC algorithm and the source code for a simplistic implementation using **Python**.

# Problem definition
Consider the distribution of points in the following diagram. 

<img src="images/Intro_HumanMind_SeesStraightLine.png"/>

The human mind can immediately spot that all the points in this distribution but for one point is aligned in a straight line and the mind has no difficulty in distinguishing the inliers from the outliers. How can me make the computer emulate this aspect of the human behavior? The RANSAC algorithm attempts to address this challenge.

# Traditional approach - Fitting a straight line using the least squares regression method
<img src="images/SimpleLinearRegression.png"/>

Consider the points above. How do we find a line which fits this distribution? One of the popular approaches is the least square distance method. In this approach we 

- Create a cost function which sum up the distance of all points from the line
- Iteratively tinker with the equation of the line and evaluate the cost function
- Select the line line which yields the lowest cost function

# How do we build a cost function?

<img src="images/y_mx_plus_c.png"/>


- Consider any point **P<sub>i</sub>** with coordinates **(X<sub>i</sub>, Y<sub>i</sub>)**
- Consider a straight line with the equation y=mx+c where  **m** is the slope and the **c** is the Y-intercept 

- The distance of P from the point where the vertical projection intersects this line is given by  **d<sub>i</sub>=(mx<sub>i</sub>+c) - y<sub>i</sub>**
- We do not want to be worried about negative values. Therefore let us square the above distance. d<sub>i</sub><sup>2</sup>=((mx<sub>i</sub>+c) - y<sub>i</sub>)<sup>2</sup>
- The summation of the square of the vertical distance of all **N** points is given by **Sum =&Sigma;(mx<sub>i</sub>+c) - y<sub>i</sub>)<sup>2</sup>**
- We can express the summation as a function which is dependent on two variables - The slope **m** and the Y intercept **c**
- The cost function **f(m,c)** can now be expressed as **&Sigma;(mx<sub>i</sub>+c) - y<sub>i</sub>)<sup>2</sup>**
- Since we have 2 variables (m and c) we need 2 equations.
- The maxima/minima of a function can be determined by using derivatives. The point where a function achieves maxima/minima the derivative of the function at that point is zero.
- We will use partial differentiation to find the values of **m** and **c** which yield the lowest value
- The partial derivatives of f(m,c) with respect to the variables m and c would have to be zero to give us the lowest cost value
- <sup>df</sup>&frasl;<sub>dm</sub> = 2&Sigma;(mx<sub>i</sub>+c) - y<sub>i</sub>)*x<sub>i</sub>
- <sup>df</sup>&frasl;<sub>dc</sub> = 2&Sigma;(mx<sub>i</sub>+c) - y<sub>i</sub>)
- In the interest of time, I will skip the derivation of the least squares distance formula and straight away present the solution
- m=(N&Sigma;(xy) -&Sigma;x&Sigma;y)/(N&Sigma;(x<sup>2</sup>) + (&Sigma;x)<sup>2</sup>)
- c= (&Sigma;y - m&Sigma;x)/N

# Challenges with least squares regression 
## Scenario - No noise
Consider the data points shown below. The data appears to follow a straight line and indeed least squares regression gives us a nice line that models the data.

<img src="images/Linear_Regression_Perfect.png"/>

## Scenario - A few noisy data points
But real data is seldom so clean. Let us add one outlier to this distribution  (10,15) and find the best fitting line using least squares regression

<img src="images/Linear_Regression_Noise.png"/>

We can see that the single outlier has brought about a considerable change. The out of box least squares method is very sensitive to outliers. 
- Algorithms like weighted least squares use the distance of a point from the straight line as a weight to minimize the impact of far flung points. 
- In this article we will restrict ourselves to RANSAC algorithm.


# Understanding RANSAC - Overview
Before getting into the full details, I have presented a distilled version of RANSAC in this section
- Randomly select a smaller set of points (**n**) from the entire distribution (**N**)
- Use least squares regression to determine the linear equation which fits the **n** points
- Determine the average of the distance of every point **N** from this line. This score can be considered to be a measure of the goodness of the line. 
- Keep track of the **score**. If this score is less than the best score obtained from previous iterations then discard the older linear equation and select the current linear equation.
- Go back the first step and continue iterating till you have completed a predetermined number of iterations
- Stop the algorithm when a predetermined number of iterations have been completed
- The linear equation available at the end of the iterations is possibly the best candidate line

We can see that the algorithm is not deterministic and hence the name *Random* in the acronym RANSAC. It is possible that you may not get the best model.


# Understanding RANSAC - Detailed
In this section I have presented the algorithm from the Wikipedia page of RANSAC
1. **MAX** = max iterations
1. **n**= number of points to pick in every iteration. Could be initialized to 2
1. **best_model** = equation of the line with best_error . Initialize to NULL
1. **best_error**= The lowest error (average distance) obtained so far. Initialize to a large number
1. **threshold_error**=if the distance of a point from a line is below this value then the point is classified as an inlier otherwise outlier
1. **threshold_inliers**=minimum number of inliers for a model to be selected
1. **k**= count of iterations completed. Initialize to 0
## Start of algorithm
- **k**=**k** + 1
- if **(k > MAX)** then stop the algorithm
- select **n** random points from entire population. Denote this set by **random_points**
- Use least square regression to find the line  which fits **random_points**. Denote this equation by **current_model**
- Determine **inliers** which is the set of all the points within **threshold_inliers** distance of **current_model**
- **count_of_inliers** =count of points in the **inliers**
- if **count_of_inliers** is less than **threshold_inliers** then abandon this sample and go to **Start**
- Extend the sample by combining **inliers** and **random_points**. 
- Use least squares regression to find the line which fits **inliers** and **random_points**. This equation is denoted as **better_model** 
- Determine the average distance of all points from **better_model**. This is denoted by **current_error**
- if (**current_error** > **best_error**) then go to **Start**
- **best_model**=**current_model**
- **best_error**=**current_error**
- Go to **Start**
## End of algorithm

# Results

## Test 1

### Input
<img src="images/NoisyImage_1.png" width="150%" height="150%"/>

### Result
<img src="images/NoisyImage_1.result.png" width="200%" height="200%"/>

## Test 2

### Input
<img src="images/NoisyImage_2.png" width="200%" height="200%"/>

### Result
<img src="images/NoisyImage_2.result.png" width="200%" height="200%"/>

## Test 3

### Input
<img src="images/NoisyImage_4.png" width="200%" height="200%"/>

### Result
<img src="images/NoisyImage_4.result.png" width="200%" height="200%"/>


# Overview of the code

## Source code location
I have used the following tools to author the Python scripts that accompany this article.
1. Visual Studio 2019
1. with Python project templates
1. Python 3.7 engine

The source code can be found at https://github.com/sdg002/RANSAC
You do not have to use Visual Studio. I am quite certain that the Phython code should work as it is.

## List of Python files and folders

- **RANSAC.py** - Outermost Python script which can be executed from the command line
- **GenerateNoisyLine.py** - Outermost Python script which will generate a random straight line with salt-pepper noise
- **LineModel.py** - Implements a class that represents the equation of a straight line
- **Point.py** - Implements a class which represents a 2d point
- **RansacHelper.py** - Implements the core RANSAC algorithm
- **Util.py** - Utility functions
- **test_??.py** - These are unit test classes
- .\input\ - The folder containing input files
- .\output\ - The folder where the resulting images are published

## Quick start - Generating an image of a noisy line
- Run the script **GenerateNoisyLine.py** to generate a rectangular image with 1 line in a random orientation and salt-pepper noise
- The resulting image will be generated in the subfolder **.\out**

<img src="images/Sample_Output_GenerateNoisyImage.66.png"/>

## Quick start - Perform RANSAC on a noisy image
- Run the script **RANSAC.py** to find the best fitting line in a noisy image
- The input file is controlled by a variable inside RANSAC.py and the this file should be placed in the subdirectory **.\input**
- The output is generated in the form of a new image which has the RANSAC line superimposed over the original line

<img src="images/Sample_Output_After_RANSAC_GenerateNoisyImage.66.png"/>

# References and further reading
- Youtube lecture (https://www.youtube.com/watch?v=BpOKB3OzQBQ)
- Wikipedia article on RANSAC (https://en.wikipedia.org/wiki/Random_sample_consensus)
- Deriving the least squares regression (https://online.stat.psu.edu/stat414/node/278/)
- Weighted least squares (https://towardsdatascience.com/when-and-how-to-use-weighted-least-squares-wls-models-a68808b1a89d)
- Hough transform (https://en.wikipedia.org/wiki/Hough_transform)
- Finding the maxima and minima (http://clas.sa.ucsb.edu/staff/lee/Max%20and%20Min's.htm)
