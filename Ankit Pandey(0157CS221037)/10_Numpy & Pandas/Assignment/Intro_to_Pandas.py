# ==========================================================
# Exercise 3: Pandas DataFrame
# ==========================================================
# Question:
# 1. Download the dataset and rename it to cars.csv
#    Dataset Links:
#    - https://www.kaggle.com/ucimi/autompg-dataset/data/select-auto-mpg.csv
#    - https://archive.ics.uci.edu/ml/datasets/Auto+MPG
#
# 2. Import Pandas
# 3. Import the Cars Dataset and store the Pandas DataFrame in a variable named 'cars'.
# 4. Inspect the first 10 rows of the DataFrame.
# 5. Print the entire DataFrame.
# 6. Inspect the last 5 rows of the DataFrame.
# 7. Get some meta information about the DataFrame using `.info()`.

# ----------------------------------------------------------
# Explanation:
# - Pandas is a library used to handle and analyze structured data.
# - `pd.read_csv()` is used to read CSV files into a DataFrame.
# - `head(10)` shows the first 10 rows of the data.
# - `tail(5)` shows the last 5 rows of the data.
# - `info()` provides a concise summary of the DataFrame,
#    including column names, non-null counts, and data types.
# ----------------------------------------------------------

import pandas as pd

# Reading the dataset
cars = pd.read_csv("cars.csv")

# 1. Inspect the first 10 rows
print("First 10 rows of the Cars DataFrame:\n", cars.head(10))

# 2. Print the entire DataFrame
print("\nFull Cars DataFrame:\n", cars)

# 3. Inspect the last 5 rows
print("\nLast 5 rows of the Cars DataFrame:\n", cars.tail(5))

# 4. Get meta information about the DataFrame
print("\nMeta Information about Cars DataFrame:")
cars.info()


# ==========================================================
# Exercise 4: 50 Startups Dataset
# ==========================================================
# Question:
# 1. Download the 50 startups dataset and save it as 50_Startups.csv
#    Dataset Link:
#    - https://www.kaggle.com/datasets/farhaned29/50-startups
#
# 2. Create a DataFrame using Pandas.
# 3. Read the data from the CSV file and load it into a DataFrame.
# 4. Check the statistical summary of the data using `.describe()`.
# 5. Check the correlation coefficient between dependent and independent variables using `.corr()`.

# ----------------------------------------------------------
# Explanation:
# - `pd.read_csv()` is used to read CSV files into Pandas DataFrames.
# - `describe()` provides statistical details like mean, median, std, etc.
# - `corr()` gives correlation between columns.
#   - Correlation close to 1 → strong positive relationship.
#   - Correlation close to -1 → strong negative relationship.
#   - Correlation near 0 → no relationship.
# ----------------------------------------------------------

# Reading the 50 startups dataset
startups = pd.read_csv("50_Startups.csv")

# 1. Display the first 5 rows to verify
print("\nFirst 5 rows of 50 Startups Dataset:\n", startups.head())

# 2. Statistical summary of the dataset
print("\nStatistical Summary of 50 Startups:\n", startups.describe())

# 3. Correlation between dependent and independent variables
print("\nCorrelation Coefficients:\n", startups.corr())
