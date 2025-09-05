# ==========================================================
# Exercise 1
# ==========================================================
# Question:
# Create a two-dimensional 3x3 array and perform the following operations:
# 1. ndim - to find the number of dimensions
# 2. shape - to find the shape of the array
# 3. slicing - to extract specific elements or rows/columns

# ----------------------------------------------------------
# Explanation:
# - A two-dimensional array is like a matrix (rows and columns).
# - `ndim` tells how many dimensions the array has.
# - `shape` returns a tuple showing the structure (rows, columns).
# - Slicing is used to get specific parts of the array like
#   arr[0] → first row,
#   arr[:, 1] → second column,
#   arr[0:2, 0:2] → top-left 2x2 sub-matrix.
# ----------------------------------------------------------

import numpy as np

# Creating a 3x3 array
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Printing the array
print("Two-Dimensional Array:\n", array_2d)

# 1. Finding number of dimensions
print("Number of Dimensions (ndim):", array_2d.ndim)

# 2. Finding shape of the array
print("Shape of the Array:", array_2d.shape)

# 3. Performing slicing
print("First Row:", array_2d[0])
print("Second Column:", array_2d[:, 1])
print("Top-Left 2x2 Sub-Matrix:\n", array_2d[0:2, 0:2])


# ==========================================================
# Exercise 2
# ==========================================================
# Question:
# Create a one-dimensional array and perform the following operations:
# 1. ndim - to find the number of dimensions
# 2. shape - to find the shape of the array
# 3. reshape - to change the shape of the array into a multi-dimensional format

# ----------------------------------------------------------
# Explanation:
# - A one-dimensional array is like a simple list.
# - `ndim` will return 1 because it's a single dimension.
# - `shape` will return (n,) where n is the number of elements.
# - `reshape` changes the structure of the array, for example:
#   reshape(2, 3) → converts 6 elements into 2 rows and 3 columns.
# ----------------------------------------------------------

# Creating a 1D array
array_1d = np.array([10, 20, 30, 40, 50, 60])

# Printing the array
print("\nOne-Dimensional Array:", array_1d)

# 1. Finding number of dimensions
print("Number of Dimensions (ndim):", array_1d.ndim)

# 2. Finding shape of the array
print("Shape of the Array:", array_1d.shape)

# 3. Reshaping the array to 2 rows and 3 columns
reshaped_array = array_1d.reshape(2, 3)
print("Reshaped Array (2x3):\n", reshaped_array)
