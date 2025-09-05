# ----------------------------------------------------------
# Diabetes Prediction - Exploratory Data Analysis (EDA)
# ----------------------------------------------------------

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Step 2: Load the data into a Pandas DataFrame
# ----------------------------------------------------------

# Load the dataset
df = pd.read_csv("Diabetes.csv")

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Show basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Shape of the dataset
print("\nShape of dataset (rows, columns):", df.shape)

# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# ----------------------------------------------------------
# Step 3: Data Preprocessing
# ----------------------------------------------------------
# 1. Check for missing values
# 2. Replace zeros with NaN in certain columns
# 3. Fill missing values using median

# Check missing values
print("\nMissing Values Before Processing:")
print(df.isnull().sum())

# Columns where 0 is considered invalid and should be treated as NaN
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN in these columns
df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

# Check missing values after replacing 0 with NaN
print("\nMissing Values After Replacing 0 with NaN:")
print(df.isnull().sum())

# Fill missing values with the median for each column
for col in columns_to_replace:
    df[col].fillna(df[col].median(), inplace=True)

# Verify that there are no missing values left
print("\nMissing Values After Filling:")
print(df.isnull().sum())

# ----------------------------------------------------------
# Step 4: Handle Categorical Data
# ----------------------------------------------------------
# The 'Outcome' column is categorical:
# 0 = No diabetes
# 1 = Diabetes

# Check unique values in 'Outcome'
print("\nUnique values in Outcome column:", df['Outcome'].unique())

# Convert Outcome to categorical data type
df['Outcome'] = df['Outcome'].astype('category')

# ----------------------------------------------------------
# Step 5: Uni-variate Analysis
# ----------------------------------------------------------
# Uni-variate analysis is done on single variables

# Histogram for all numerical columns
df.hist(figsize=(12, 10), bins=20, color='teal')
plt.suptitle("Histograms of Numerical Columns", fontsize=16)
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, palette="Set2")
plt.title("Boxplots for Outlier Detection")
plt.xticks(rotation=45)
plt.show()

# Count plot for categorical column 'Outcome'
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette='viridis')
plt.title("Count of Diabetes Outcome")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()

# ----------------------------------------------------------
# Step 6: Bi-variate Analysis
# ----------------------------------------------------------
# Bi-variate analysis checks relationships between two variables

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Pairplot to visualize pairwise relationships between features
sns.pairplot(df, hue='Outcome', diag_kind='kde', palette='husl')
plt.suptitle("Pairplot of Features vs Outcome", y=1.02)
plt.show()

# Scatter plot example: Glucose vs BMI
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, palette='coolwarm')
plt.title("Glucose vs BMI by Outcome")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.show()

# ----------------------------------------------------------
# Step 7: Summary
# ----------------------------------------------------------
# In this notebook we:
# 1. Loaded the data into a DataFrame
# 2. Handled missing and inappropriate data
# 3. Converted categorical data properly
# 4. Performed Uni-variate and Bi-variate analysis
# ----------------------------------------------------------
