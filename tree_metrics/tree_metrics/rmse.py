import argparse

import numpy as np


# Root Mean Squared Error
def rmse(predictions, targets):
    # predictions and targets are numpy arrays
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

# usage example:

# predictions = np.array([[4.0, 2.7, 3.2], [1.2, 3.4, 2.1]])
# targets = np.array([[3.0, 2.0, 3.0], [1.0, 3.5, 2.2]])
# print(rmse(predictions, targets))  # Outputs: 0.5147815070493501

# use this as a standalone script to compute the RMSE of two arrays
if __name__ == "__main__":
 

    parser = argparse.ArgumentParser(description="Compute the RMSE of two arrays.")
    parser.add_argument("predictions_file", help="Path to predictions file.")
    parser.add_argument("targets_file", help="Path to targets file.")
    args = parser.parse_args()

    # the input files should be text files containing numpy arrays
    predictions = np.loadtxt(args.predictions_file)
    targets = np.loadtxt(args.targets_file)

    print(rmse(predictions, targets))