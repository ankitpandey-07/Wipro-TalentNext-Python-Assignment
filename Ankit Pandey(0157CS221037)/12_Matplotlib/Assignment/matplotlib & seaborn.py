# ==========================================================
# Exploratory Data Analysis (EDA)
# ==========================================================
# Datasets:
# 1. Mall Customers Dataset
# 2. Salary Data Dataset
# 3. Social Network Ads Dataset
#
# Libraries Used:
# - Pandas for data handling
# - Matplotlib and Seaborn for visualization
# ==========================================================

# ==========================
# Step 1: Import Libraries
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Enable inline plotting
#matplotlib inline   (does not work in vs code )

# Apply Seaborn theme for better visuals
sns.set(style="whitegrid")

# ==========================================================
# EDA 1: Mall Customers Dataset
# ==========================================================
# Question:
# Perform EDA on Mall Customers dataset using Matplotlib and Seaborn.
# ==========================================================

# Step 1: Load dataset
mall_df = pd.read_csv("Mall_Customers.csv")

# Step 2: Basic info
print("Mall Customers Dataset Shape:", mall_df.shape)
print("\nFirst 5 rows:\n", mall_df.head())
print("\nSummary Statistics:\n", mall_df.describe())

# Step 3: Check missing values
print("\nMissing Values:\n", mall_df.isnull().sum())

# Step 4: Visualizations
# 4.1 Distribution of Age
plt.figure(figsize=(8, 5))
sns.histplot(mall_df['Age'], kde=True, bins=20, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 4.2 Gender Count Plot
plt.figure(figsize=(6, 5))
sns.countplot(x='Gender', data=mall_df, palette='viridis')
plt.title('Gender Count')
plt.show()

# 4.3 Income vs Spending Score
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Gender', data=mall_df, palette='coolwarm')
plt.title('Income vs Spending Score')
plt.show()

# ==========================================================
# EDA 2: Salary Data Dataset
# ==========================================================
# Question:
# Perform EDA on Salary dataset using Matplotlib and Seaborn.
# ==========================================================

# Step 1: Load dataset
salary_df = pd.read_csv("Salary_Data.csv")

# Step 2: Basic info
print("\nSalary Dataset Shape:", salary_df.shape)
print("\nFirst 5 rows:\n", salary_df.head())
print("\nSummary Statistics:\n", salary_df.describe())

# Step 3: Missing values
print("\nMissing Values:\n", salary_df.isnull().sum())

# Step 4: Visualizations
# 4.1 Years of Experience vs Salary
plt.figure(figsize=(8, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=salary_df, color='orange')
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# 4.2 Regression line (trend)
plt.figure(figsize=(8, 6))
sns.regplot(x='YearsExperience', y='Salary', data=salary_df, scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.title('Salary Trend by Experience')
plt.show()

# ==========================================================
# EDA 3: Social Network Ads Dataset
# ==========================================================
# Question:
# Perform EDA on Social Network Ads dataset using Matplotlib and Seaborn.
# ==========================================================

# Step 1: Load dataset
ads_df = pd.read_csv("Social_Network_Ads.csv")

# Step 2: Basic info
print("\nSocial Network Ads Dataset Shape:", ads_df.shape)
print("\nFirst 5 rows:\n", ads_df.head())
print("\nSummary Statistics:\n", ads_df.describe())

# Step 3: Missing values
print("\nMissing Values:\n", ads_df.isnull().sum())

# Step 4: Visualizations
# 4.1 Gender distribution
plt.figure(figsize=(6, 5))
sns.countplot(x='Gender', data=ads_df, palette='pastel')
plt.title('Gender Distribution')
plt.show()

# 4.2 Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(ads_df['Age'], kde=True, bins=15, color='green')
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 4.3 Age vs Estimated Salary
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=ads_df, palette='coolwarm')
plt.title('Age vs Estimated Salary (Purchased Highlighted)')
plt.show()

# ==========================================================
# Final Notes:
# ==========================================================
# - We visualized key patterns in three different datasets.
# - Checked missing values, distributions, and correlations.
# - Plots help in understanding data before modeling.
# ==========================================================
