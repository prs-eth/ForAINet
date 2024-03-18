import math
import numpy as np
#
#Generate a normally distributed random cluster of points around x,y
#Takes in a point x,y and the standard deviation (stddev) 
#and the number of points to generate (num)
#
def GenerateClusterOfRandomPointsAroundXY(x,y,stddev,num):
    angleStart=0
    angleEnd=2*3.1415
    #angleStepDegrees=10
    #angleStepRadians=2*3.1415/360 * angleStepDegrees
    #num=12
    angles=np.linspace(angleStart,angleEnd,num)
    mean=0
    random_radii=np.random.normal(mean, stddev, len(angles)) 
    np_results=np.zeros((len(angles),2))
    for idx in range(0,len(angles)):
        theta=angles[idx]
        radii=abs(random_radii[idx])
        random_x=radii* math.cos(theta)  +x
        random_y=radii* math.sin(theta)  +y
        #print("angle=%f,x,y=%f,%f" % (theta,random_x,random_y))
        np_results[idx][0]=random_x
        np_results[idx][1]=random_y

    return np_results

