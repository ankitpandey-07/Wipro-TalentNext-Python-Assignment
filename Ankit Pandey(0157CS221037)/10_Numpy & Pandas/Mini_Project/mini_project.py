# ==========================================================
# Use Case: Outlier Detection
# ==========================================================
# Question:
# 1. Load the given dataset (datasetExample.csv) into a Pandas DataFrame.
# 2. Perform outlier detection on the dataset to identify unusual values.
#
# ----------------------------------------------------------
# Explanation:
# - **Outliers** are data points that differ significantly from other observations.
# - They can occur due to errors in data collection or represent rare events.
# - Methods to detect outliers:
#   1. **Boxplot Visualization** → Outliers appear as points outside the whiskers.
#   2. **Z-Score Method** → Values with Z-score > 3 or < -3 are considered outliers.
#   3. **IQR (Interquartile Range) Method** → Common statistical technique.
#
#   IQR Method Steps:
#   - Q1 = 25th percentile
#   - Q3 = 75th percentile
#   - IQR = Q3 - Q1
#   - Lower Bound = Q1 - 1.5 * IQR
#   - Upper Bound = Q3 + 1.5 * IQR
#   - Any value < Lower Bound or > Upper Bound is an outlier.
#
# ----------------------------------------------------------

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("datasetExample.csv")

# Display first few rows of the dataset
print("First 5 Rows of the Dataset:\n", df.head())

# Step 2: Basic information about the dataset
print("\nDataset Info:")
df.info()

print("\nStatistical Summary:\n", df.describe())

# ----------------------------------------------------------
# Step 3: Visual Outlier Detection using Boxplot
# ----------------------------------------------------------
# Plot boxplot for each numerical column to visually inspect outliers
print("\nGenerating Boxplots for Numerical Columns...")
df.plot(kind='box', subplots=True, layout=(2, 3), figsize=(12, 8), sharex=False, sharey=False)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Step 4: Detect Outliers using IQR Method
# ----------------------------------------------------------
print("\nDetecting Outliers using IQR Method:")

# Select only numerical columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)  # 25th percentile
    Q3 = df[col].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1

    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nColumn: {col}")
    if not outliers.empty:
        print(f"Outliers Detected:\n{outliers[[col]]}")
    else:
        print("No outliers detected in this column.")

# ----------------------------------------------------------
# Step 5: Summary
# ----------------------------------------------------------
# This method helps identify unusual values that may need cleaning
# or special attention before further analysis.
# ----------------------------------------------------------
