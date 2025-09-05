# ======================================================================
# TEXT PREPROCESSING FOR SMSSpamCollection DATASET
# ======================================================================
# Objective:
# The goal of this script is to clean and preprocess SMS messages so they
# can be used for building a machine learning model for spam detection.
#
# Steps Involved:
# 1. Load the dataset into a Pandas DataFrame.
# 2. Perform basic cleaning:
#    - Convert labels (ham/spam) to numeric values.
#    - Convert text to lowercase.
# 3. Remove punctuation, numbers, and special characters.
# 4. Tokenize the text (split into words).
# 5. Remove stopwords like 'the', 'is', etc.
# 6. Apply stemming to reduce words to their base form.
# 7. Join final cleaned tokens back into processed text.
# 8. Save the preprocessed data into a CSV file for future use.
# ======================================================================

# ----------------------------------------------------------
# Step 1: Import Required Libraries
# ----------------------------------------------------------
import pandas as pd        # For handling datasets
import numpy as np         # For numerical operations
import string              # For handling punctuation
import re                  # For regular expressions
import nltk                # For text preprocessing

# Download necessary NLTK packages (only required the first time)
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords  # List of stopwords
from nltk.stem import PorterStemmer  # For stemming words
from nltk.tokenize import word_tokenize  # For tokenization

# ----------------------------------------------------------
# Step 2: Load the Dataset
# ----------------------------------------------------------
# The dataset is tab-separated and contains two columns:
# 'label' -> ham or spam
# 'message' -> actual SMS text

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# Display the first 5 rows to verify data is loaded correctly
print("First 5 rows of the dataset:")
print(df.head())

# Get general information about the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())

# ----------------------------------------------------------
# Step 3: Basic Data Cleaning
# ----------------------------------------------------------
# Explanation:
# - Convert labels to numeric form for machine learning models.
#   'ham' = 0 (not spam), 'spam' = 1 (spam)
# - Convert messages to lowercase for uniformity.

# Convert labels to numerical values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Convert messages to lowercase
df['message'] = df['message'].str.lower()

print("\nAfter converting labels and making text lowercase:")
print(df.head())

# ----------------------------------------------------------
# Step 4: Remove Punctuation, Numbers, and Special Characters
# ----------------------------------------------------------
# Explanation:
# - Remove all characters except alphabets.
# - This helps clean the text and focus on meaningful words.

def clean_text(text):
    # Replace anything that is NOT a letter with a space
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()  # Convert again to lowercase
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Apply cleaning function to each message
df['clean_message'] = df['message'].apply(clean_text)

print("\nSample of cleaned messages:")
print(df[['message', 'clean_message']].head())

# ----------------------------------------------------------
# Step 5: Tokenization
# ----------------------------------------------------------
# Explanation:
# - Tokenization splits sentences into individual words.
# - Example: "free entry in competition" -> ['free', 'entry', 'in', 'competition']

def tokenize_text(text):
    return word_tokenize(text)

# Apply tokenization
df['tokens'] = df['clean_message'].apply(tokenize_text)

print("\nTokenized messages:")
print(df[['clean_message', 'tokens']].head())

# ----------------------------------------------------------
# Step 6: Remove Stopwords
# ----------------------------------------------------------
# Explanation:
# - Stopwords are common words like 'the', 'and', 'is', etc.
# - These words do not add much meaning and can be removed to reduce noise.

stop_words = set(stopwords.words('english'))  # Get list of English stopwords

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Apply stopword removal
df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)

print("\nAfter removing stopwords:")
print(df[['tokens', 'tokens_no_stopwords']].head())

# ----------------------------------------------------------
# Step 7: Stemming
# ----------------------------------------------------------
# Explanation:
# - Stemming reduces words to their root/base form.
# - Example:
#   'running' -> 'run'
#   'studies' -> 'studi'

ps = PorterStemmer()

def stem_words(tokens):
    return [ps.stem(word) for word in tokens]

# Apply stemming
df['final_tokens'] = df['tokens_no_stopwords'].apply(stem_words)

print("\nAfter applying stemming:")
print(df[['tokens_no_stopwords', 'final_tokens']].head())

# ----------------------------------------------------------
# Step 8: Join Final Tokens Back to Text
# ----------------------------------------------------------
# Explanation:
# - After cleaning, tokenizing, removing stopwords, and stemming,
#   we join the list of words back into a single string for modeling.

df['processed_message'] = df['final_tokens'].apply(lambda x: " ".join(x))

print("\nFinal processed messages:")
print(df[['message', 'processed_message']].head())

# ----------------------------------------------------------
# Step 9: Save the Preprocessed Data (Optional)
# ----------------------------------------------------------
# Save the preprocessed dataset into a new CSV file for later use
df.to_csv('processed_sms.csv', index=False)
print("\nProcessed data saved as 'processed_sms.csv'")

# ----------------------------------------------------------
# Step 10: Summary of Preprocessing
# ----------------------------------------------------------
# 1. Loaded dataset and checked for missing values.
# 2. Converted labels to numeric form.
# 3. Converted text to lowercase for uniformity.
# 4. Removed punctuation, numbers, and special characters.
# 5. Tokenized messages into individual words.
# 6. Removed stopwords to reduce noise.
# 7. Applied stemming to reduce words to base forms.
# 8. Joined cleaned tokens back into text form.
# 9. Saved the final preprocessed dataset into a CSV file.
# ----------------------------------------------------------

# Final DataFrame Columns:
# - label              -> 0 = ham, 1 = spam
# - message            -> Original message
# - clean_message      -> Message after removing noise
# - tokens             -> List of words
# - tokens_no_stopwords-> Tokens without stopwords
# - final_tokens       -> Stemmed tokens
# - processed_message  -> Final text ready for modeling
