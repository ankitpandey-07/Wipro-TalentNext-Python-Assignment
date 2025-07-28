# Q1- Write a program to check if a string has only octal digits (0–7).
# Given list: ['789','123','004']

import re
strings = ['789', '123', '004']
for s in strings:
    if re.fullmatch(r'[0-7]+', s):
        print(f"'{s}' → Valid Octal")
    else:
        print(f"'{s}' → Invalid Octal")

# Q2- Extract the user id, domain name, and suffix from the following emails:
# emails = """zuck@facebook.com
# sunder33@google.com
# jeff42@amazon.com"""

emails = """zuck@facebook.com
sunder33@google.com
jeff42@amazon.com"""
pattern = r'(\w+)@(\w+)\.(\w+)'
matches = re.findall(pattern, emails)
for user, domain, suffix in matches:
    print(f"User ID: {user}, Domain: {domain}, Suffix: {suffix}")

# Q3-Split the given irregular sentence into proper words.
#sentence = """A, very    very; irregular_sentence"""
# Expected output:
# A very very irregular sentence

sentence = """A, very    very; irregular_sentence"""
cleaned = re.sub(r'[,;_]', ' ', sentence)
words = re.split(r'\s+', cleaned.strip())
result = ' '.join(words)
print(result)

#Q4- Clean up the following tweet so that it contains only the user’s message.
#Remove all URLs, hashtags, RTs, and mentions/CCs.
#tweet = '''Good advice! RT @TheNextWeb: What I would do differently if I was learning to code today http://t.co/lbwej0pxOd cc: @garybernhardt #rstats'''
# Expected output:
# Good advice! What I would do differently if I was learning to code today

tweet = '''Good advice! RT @TheNextWeb: What I would do differently if I was learning to code today http://t.co/lbwej0pxOd cc: @garybernhardt #rstats'''
tweet = re.sub(r'\bRT\b', '', tweet)
tweet = re.sub(r'http\S+', '', tweet)
tweet = re.sub(r'@\w+', '', tweet)
tweet = re.sub(r'#\w+', '', tweet)
tweet = re.sub(r'\bcc:?', '', tweet)
cleaned = re.sub(r'\s+', ' ', tweet).replace(':', '').strip()
print(cleaned)

# Q5- Extract all the text portions between the tags from the following HTML page:
# https://raw.githubusercontent.com/selva86/datasets/master/sample.html

from bs4 import BeautifulSoup
import requests
r = requests.get("https://raw.githubusercontent.com/selva86/datasets/master/sample.html")
soup = BeautifulSoup(r.text, 'html.parser')
output = [element.get_text() for element in soup.find_all()]
print(output)

# Q6- Given below list of words, identify the words that begin and end with the same character.

words = ['civic', 'trust', 'widows', 'maximum', 'museums', 'aa', 'as']
matching_words = [word for word in words if word[0] == word[-1]]
print(matching_words)
