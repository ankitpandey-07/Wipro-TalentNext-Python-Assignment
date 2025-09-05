# ======================================================================
# Use Case 1: Predict Cancer Based on Features
# Dataset: cancer.csv
# Topics Covered: Logistic Regression and Evaluation Metrics
# ======================================================================

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Dataset
cancer_df = pd.read_csv("cancer.csv")

# Display first 5 rows
print("First 5 rows of the Cancer Dataset:")
print(cancer_df.head())

# Step 3: Data Preprocessing
# Check for missing values
print("\nMissing values in the dataset:")
print(cancer_df.isnull().sum())

# Drop any rows with missing values (if present)
cancer_df.dropna(inplace=True)

# Identify target and features
# Assuming 'Outcome' or 'diagnosis' column is the target
target_column = 'diagnosis'  # Change this name if your dataset has a different column
X = cancer_df.drop(target_column, axis=1)  # Independent variables
y = cancer_df[target_column]               # Dependent variable

# If target is categorical like 'M' or 'B', convert to binary 0/1
if y.dtype == 'object':
    y = y.map({'M': 1, 'B': 0})

# Step 4: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
log_model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = log_model.predict(X_test)

# Step 7: Evaluate the Model
print("\nCancer Prediction Model Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Cancer Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ======================================================================
# Use Case 2: Predict Customer Purchase Based on Features
# Dataset: Social_Network_Ads.csv
# Topics Covered: Logistic Regression and Evaluation Metrics
# ======================================================================

# Step 1: Load the Dataset
ads_df = pd.read_csv("Social_Network_Ads.csv")

# Display first 5 rows
print("\nFirst 5 rows of Social Network Ads Dataset:")
print(ads_df.head())

# Step 2: Data Preprocessing
# Check for missing values
print("\nMissing values in the dataset:")
print(ads_df.isnull().sum())

# Target column is usually 'Purchased'
target_column = 'Purchased'
X = ads_df.drop(target_column, axis=1)  # Independent features
y = ads_df[target_column]               # Dependent variable

# Convert categorical columns (like Gender) to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Step 3: Split the Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the Logistic Regression Model
log_ads_model = LogisticRegression(max_iter=1000)
log_ads_model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred_ads = log_ads_model.predict(X_test)

# Step 6: Evaluate the Model
print("\nCustomer Purchase Prediction Model Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred_ads))

# Confusion Matrix
cm_ads = confusion_matrix(y_test, y_pred_ads)
print("\nConfusion Matrix:\n", cm_ads)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_ads))

# Step 7: Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_ads, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Customer Purchase Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ======================================================================
# Summary
# ======================================================================
# 1. Loaded both datasets (cancer.csv and Social_Network_Ads.csv)
# 2. Performed preprocessing:
#    - Handled missing values
#    - Converted categorical data into numeric format
# 3. Built and trained Logistic Regression models
# 4. Evaluated models using:
#    - Accuracy Score
#    - Confusion Matrix
#    - Classification Report
# 5. Visualized results using Seaborn heatmaps
# ======================================================================
