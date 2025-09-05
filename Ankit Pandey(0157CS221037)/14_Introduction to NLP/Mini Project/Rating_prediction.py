# ===============================================
# Step 1: Import Required Libraries
# ===============================================
# pandas - For data handling
# numpy - For numerical operations
# matplotlib & seaborn - For visualizations
# sklearn - For ML model building and evaluation
# nltk - For text preprocessing
# re - For regular expressions (cleaning text)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================================
# Step 2: Load the Dataset
# ===============================================
# Load the yelp.csv file into a pandas DataFrame
# Ensure that 'yelp.csv' is in the same directory OR provide the full path
df = pd.read_csv("yelp.csv")

# Display the first 5 rows to understand the structure of the dataset
print(df.head())

# ===============================================
# Step 3: Explore the Dataset
# ===============================================
# Checking basic info to understand data types and missing values
print(df.info())

# Checking for null values
print(df.isnull().sum())

# Checking distribution of target variable (Stars)
sns.countplot(x='stars', data=df)
plt.title("Distribution of Star Ratings")
plt.show()

# ===============================================
# Step 4: Data Preprocessing
# ===============================================

# --- Step 4.1: Handle Missing Values ---
# Drop rows with missing values, if any
df.dropna(inplace=True)

# --- Step 4.2: Text Cleaning Function ---
# Define a function to clean review text:
# - Remove special characters, numbers
# - Convert text to lowercase
# - Remove extra spaces

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only alphabets and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Check the cleaned text
print(df[['text', 'cleaned_text']].head())

# ===============================================
# Step 5: Text Vectorization
# ===============================================
# Machine learning models cannot directly work with text,
# so we need to convert text into numerical features.

# We will use TF-IDF (Term Frequency - Inverse Document Frequency)

vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 words
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Define the target variable (Star ratings)
y = df['stars']

# ===============================================
# Step 6: Train-Test Split
# ===============================================
# Split the data into training and testing sets (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================================
# Step 7: Build the Model
# ===============================================
# We will use Logistic Regression for multi-class classification

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================================
# Step 8: Make Predictions
# ===============================================
y_pred = model.predict(X_test)

# ===============================================
# Step 9: Model Evaluation
# ===============================================

# --- 9.1: Accuracy Score ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# --- 9.2: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- 9.3: Classification Report ---
# Provides precision, recall, and F1-score for each class
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ===============================================
# Step 10: Testing the Model with New Input
# ===============================================
# Let's test the model with a custom review

sample_review = ["The food was absolutely wonderful, service was fantastic!"]
sample_cleaned = [clean_text(sample_review[0])]
sample_vectorized = vectorizer.transform(sample_cleaned).toarray()

predicted_star = model.predict(sample_vectorized)
print(f"Predicted Rating for sample review: {predicted_star[0]}")

