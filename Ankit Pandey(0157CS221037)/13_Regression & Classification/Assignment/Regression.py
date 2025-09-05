# ======================================================================
# Use Case 1: Predict the Price of a Car Based on its Features
# Dataset: cars.csv
# Topics Covered: Multiple Linear Regression
# ======================================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
cars = pd.read_csv("cars.csv")

# Display first 5 rows
print("First 5 rows of the cars dataset:")
print(cars.head())

# Step 3: Data Preprocessing
# Check for missing values
print("\nMissing values in the dataset:")
print(cars.isnull().sum())

# Drop rows with missing values (if any)
cars.dropna(inplace=True)

# Check data types and convert if needed
print("\nDataset info after cleaning:")
print(cars.info())

# Step 4: Define Features and Target
# Example: Predicting 'Price' based on other features
X = cars.drop('Price', axis=1)  # Independent variables
y = cars['Price']               # Dependent variable

# Convert categorical data using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Step 5: Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
print("\nCar Price Prediction Model Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Visualize Actual vs Predicted Prices
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()


# ======================================================================
# Use Case 2: Predict the Profit of a Startup Based on its Features
# Dataset: 50_Startups.csv
# Topics Covered: Multiple Linear Regression
# ======================================================================

# Step 1: Load the dataset
startups = pd.read_csv("50_Startups.csv")

# Display first 5 rows
print("\nFirst 5 rows of 50_Startups dataset:")
print(startups.head())

# Step 2: Check for missing values
print("\nMissing values in the dataset:")
print(startups.isnull().sum())

# Step 3: Separate features and target
X = startups.drop('Profit', axis=1)
y = startups['Profit']

# Convert categorical data (State column) into dummy variables
X = pd.get_dummies(X, drop_first=True)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Model
model_startup = LinearRegression()
model_startup.fit(X_train, y_train)

# Step 6: Predict using the model
y_pred_startup = model_startup.predict(X_test)

# Step 7: Evaluate the model
print("\nStartup Profit Prediction Model Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_startup))
print("R2 Score:", r2_score(y_test, y_pred_startup))

# Step 8: Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_startup, color='green')
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Startup Profits")
plt.show()


# ======================================================================
# Use Case 3: Predict the Salary Based on Experience
# Dataset: Salary_data.csv
# Topics Covered: Simple Linear Regression
# ======================================================================

# Step 1: Load the dataset
salary = pd.read_csv("Salary_data.csv")

# Display first 5 rows
print("\nFirst 5 rows of Salary dataset:")
print(salary.head())

# Step 2: Check for missing values
print("\nMissing values in the dataset:")
print(salary.isnull().sum())

# Step 3: Define features and target
X = salary[['YearsExperience']]  # Independent variable
y = salary['Salary']            # Dependent variable

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and train the model
model_salary = LinearRegression()
model_salary.fit(X_train, y_train)

# Step 6: Predict using the model
y_pred_salary = model_salary.predict(X_test)

# Step 7: Evaluate the model
print("\nSalary Prediction Model Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_salary))
print("R2 Score:", r2_score(y_test, y_pred_salary))

# Step 8: Visualize the regression line
plt.figure(figsize=(8,6))
sns.scatterplot(x=X, y=y, color='red', label="Actual Data")
plt.plot(X, model_salary.predict(X), color='blue', label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience vs Salary")
plt.legend()
plt.show()
