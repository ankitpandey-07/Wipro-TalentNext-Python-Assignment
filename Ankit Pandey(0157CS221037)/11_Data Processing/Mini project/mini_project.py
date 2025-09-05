# ==========================================================
# Use-Case: House Price Prediction - Data Preprocessing
# ==========================================================
# Dataset: melb_data.csv
# Kaggle Link: https://www.kaggle.com/datasets/gunjanpathak/melb-data
#
# Tasks:
# 1. Load the data in dataframe (Pandas)
# 2. Handle inappropriate data
# 3. Handle the missing data
# 4. Handle the categorical data
# ==========================================================

# ==========================
# Step 1: Import Libraries
# ==========================
import pandas as pd
import numpy as np

# ==========================
# Step 2: Load the Dataset
# ==========================
# Explanation:
# - We use pandas `read_csv()` to load the CSV file into a dataframe.
# - This allows easy manipulation and cleaning of the data.

df = pd.read_csv("melb_data.csv")

# Display initial structure
print("Initial Shape of Dataset:", df.shape)
print("\nFirst 5 Rows of Dataset:\n", df.head())

# ==========================
# Step 3: Handle Inappropriate Data
# ==========================
# Explanation:
# - Inappropriate data includes incorrect data types or irrelevant columns.
# - Example: Columns like 'Address', 'SellerG' are identifiers and do not help in prediction.
# - We also ensure numeric columns have the correct data type.

# Dropping irrelevant columns
irrelevant_columns = ['Address', 'SellerG', 'Date']
df.drop(columns=irrelevant_columns, inplace=True, errors='ignore')

# Fixing incorrect data types if needed
# For example, if 'YearBuilt' is stored as object instead of int
if df['YearBuilt'].dtype == 'object':
    df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce')

print("\nData Types After Fixing:\n", df.dtypes)

# ==========================
# Step 4: Handle Missing Data
# ==========================
# Explanation:
# - Missing values need to be filled or dropped depending on context.
# - Strategy:
#   * For numerical columns → Fill with Median (robust to outliers)
#   * For categorical columns → Fill with Mode (most common value)

# Checking missing data
print("\nMissing Values Before Handling:\n", df.isnull().sum())

# Filling missing values
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column].fillna(df[column].median(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)

# Verify missing values handled
print("\nMissing Values After Handling:\n", df.isnull().sum())

# ==========================
# Step 5: Handle Categorical Data
# ==========================
# Explanation:
# - Machine learning models require numerical data.
# - We use One-Hot Encoding to convert categorical variables to numeric form.
# - This will create binary columns for each category.

categorical_columns = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns Before Encoding:\n", categorical_columns)

df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("\nShape After Encoding:", df_encoded.shape)
print("\nEncoded Data Preview:\n", df_encoded.head())

# ==========================
# Step 6: Final Cleaned Data
# ==========================
# Save the preprocessed dataset for future use
df_encoded.to_csv("melb_data_cleaned_for_model.csv", index=False)
print("\nFinal cleaned dataset saved as 'melb_data_cleaned_for_model.csv'")
