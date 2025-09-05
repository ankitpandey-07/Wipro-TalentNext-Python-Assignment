# ======================================================================
# Diabetes Prediction Model
# Dataset: PIMA Indians Diabetes Dataset
# Models: Logistic Regression and K-Nearest Neighbors
# ======================================================================

# ----------------------------------------------------------
# Step 1: Import Required Libraries
# ----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------------------------------------
# Step 2: Load the Dataset
# ----------------------------------------------------------
# Make sure you have the 'diabetes.csv' file in your working directory
df = pd.read_csv("diabetes.csv")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Basic info
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# ----------------------------------------------------------
# Step 3: Data Preprocessing
# ----------------------------------------------------------
# 1. Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 2. Replace zeros with NaN in certain columns where 0 is invalid
# Columns like Glucose, BloodPressure, SkinThickness, Insulin, BMI cannot be zero
invalid_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[invalid_columns] = df[invalid_columns].replace(0, np.nan)

# Check again for missing values after replacement
print("\nMissing values after replacing zeros:")
print(df.isnull().sum())

# 3. Fill missing values using median
for col in invalid_columns:
    df[col] = df[col].fillna(df[col].median())

# Verify missing values are handled
print("\nMissing values after imputation:")
print(df.isnull().sum())

# ----------------------------------------------------------
# Step 4: Exploratory Data Analysis (EDA)
# ----------------------------------------------------------

# 1. Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title("Diabetes Outcome Distribution (0 = No Diabetes, 1 = Diabetes)")
plt.show()

# 2. Histograms of features
df.hist(figsize=(15,10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------------------------
# Step 5: Define Features (X) and Target (y)
# ----------------------------------------------------------
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target variable

# Standardize the feature data for KNN and Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# Step 6: Split the Dataset
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining and Testing Set Shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# ----------------------------------------------------------
# Step 7: Build and Train Logistic Regression Model
# ----------------------------------------------------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Predict
y_pred_log = log_model.predict(X_test)

# ----------------------------------------------------------
# Step 8: Build and Train K-Nearest Neighbor Model
# ----------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)  # k=5
knn_model.fit(X_train, y_train)

# Predict
y_pred_knn = knn_model.predict(X_test)

# ----------------------------------------------------------
# Step 9: Evaluate the Models
# ----------------------------------------------------------

def evaluate_model(model_name, y_test, y_pred):
    print(f"\n===== {model_name} Evaluation =====")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Logistic Regression Evaluation
evaluate_model("Logistic Regression", y_test, y_pred_log)

# KNN Evaluation
evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn)

# ----------------------------------------------------------
# Step 10: Compare Models
# ----------------------------------------------------------
accuracy_log = accuracy_score(y_test, y_pred_log)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_log:.4f}")
print(f"KNN Accuracy: {accuracy_knn:.4f}")

# ----------------------------------------------------------
# Step 11: Visualize Confusion Matrices
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# KNN Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("KNN Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Step 12: Summary of Steps
# ----------------------------------------------------------
# 1. Loaded and inspected the dataset
# 2. Handled missing/invalid values using median imputation
# 3. Performed Exploratory Data Analysis (EDA)
# 4. Standardized features for model training
# 5. Built and evaluated Logistic Regression and KNN models
# 6. Compared performance using accuracy and classification metrics
# ----------------------------------------------------------
