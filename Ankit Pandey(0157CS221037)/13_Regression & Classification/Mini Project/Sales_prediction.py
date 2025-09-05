# ======================================================================
# Sales Prediction Based on Campaigning Expenses
# Dataset: Advertising.csv
# Topics Covered: Data Preprocessing, EDA, Multiple Linear Regression, Evaluation
# ======================================================================

# ----------------------------------------------------------
# Step 1: Import Required Libraries
# ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# Step 2: Load the Data
# ----------------------------------------------------------
# Load dataset into DataFrame
df = pd.read_csv("Advertising.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Display basic statistics
print("\nSummary Statistics:")
print(df.describe())

# ----------------------------------------------------------
# Step 3: Data Preprocessing
# ----------------------------------------------------------
# 1. Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# If there are missing values, you can fill or drop them (Here we just show the step)
df.dropna(inplace=True)  # Dropping missing values for simplicity

# 2. Check for duplicate rows
print("\nNumber of duplicate rows before removing:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Number of duplicate rows after removing:", df.duplicated().sum())

# 3. Handle categorical data (if any)
# If there's a categorical column like 'Region' or 'Channel', convert it using one-hot encoding
# Example:
# df = pd.get_dummies(df, drop_first=True)

# ----------------------------------------------------------
# Step 4: Exploratory Data Analysis (EDA)
# ----------------------------------------------------------
# Visualizing the relationships between independent variables and sales

# Histogram of each feature
df.hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Dataset Features", fontsize=16)
plt.show()

# Pairplot to check relationships
sns.pairplot(df)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------------------------
# Step 5: Define Features and Target
# ----------------------------------------------------------
# Assuming the dataset has columns: TV, Radio, Newspaper, Sales
# Target column = 'Sales', others are independent variables

X = df.drop('Sales', axis=1)  # Features
y = df['Sales']               # Target variable

# ----------------------------------------------------------
# Step 6: Split the Dataset into Train and Test Sets
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining and testing dataset shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# ----------------------------------------------------------
# Step 7: Build and Train the Multiple Linear Regression Model
# ----------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------------
# Step 8: Make Predictions
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# Step 9: Evaluate the Model
# ----------------------------------------------------------
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# R² Score
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# ----------------------------------------------------------
# Step 10: Visualize Actual vs Predicted Sales
# ----------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)  # Line of perfect prediction
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ----------------------------------------------------------
# Step 11: Summary of Steps
# ----------------------------------------------------------
# 1. Loaded dataset and checked for missing values and duplicates
# 2. Performed EDA:
#    - Histograms
#    - Pairplot
#    - Correlation heatmap
# 3. Handled categorical data (if present)
# 4. Built a Multiple Linear Regression model
# 5. Evaluated model using MSE and R² Score
# 6. Visualized actual vs predicted sales
# ----------------------------------------------------------
