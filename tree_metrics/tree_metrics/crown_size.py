import argparse

import numpy as np
from scipy.spatial import ConvexHull


def crown_size(points):
    """
    Compute the crown size of a point cloud as the area of the smallest 2D convex hull.

    Parameters:
    points (np.array): a numpy array of shape (N, 3) representing the point cloud,
                       where N is the number of points, and each point is represented
                       by its (x, y, z) coordinates.

    Returns:
    float: the area of the smallest 2D convex hull.
    """
    # project points onto the x-y plane

    points_2d = points[:, :2]  # extract x and y coordinates

    hull = ConvexHull(points_2d)

    return hull.volume # return the area of the convex hull

# Example usage:
# points = np.random.rand(30, 3)  # generate 30 random points in 3-D space
# print("Crown size: ", crown_size(points))

# use this as a standalone script to compute the crown size of a point cloud
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute the crown size of a point cloud.")
    parser.add_argument("input_file", help="Path to input file containing point cloud.")
    args = parser.parse_args()

    # the input file should be a text file containing a numpy array of shape (N, 3)
    points = np.loadtxt(args.input_file)

    print("Crown size: ", crown_size(points))

