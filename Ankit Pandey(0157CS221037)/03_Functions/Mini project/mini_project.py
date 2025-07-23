'''🔹 Project 1
Sort hyphen-separated colors alphabetically'''
def sort_colors(color_string):
    colors = color_string.split('-')
    colors.sort()
    return '-'.join(colors)

# Sample Test
print(sort_colors("green-red-yellow-black-white"))   # Output: black-green-red-white-yellow
print(sort_colors("PINK-BLUE-TAN-PURPLE"))           # Output: BLUE-PINK-PURPLE-TAN

# 🔹 Project 2: mymodule.py
"""
This module provides:
- ispalindrome(name): Checks if a string is a palindrome.
- count_the_vowels(name): Counts vowels in the string.
- frequency_of_letters(name): Returns frequency of each character.
"""

def ispalindrome(name):
    if name == name[::-1]:
        return "Yes it is a palindrome."
    else:
        return "No it is not a palindrome."

def count_the_vowels(name):
    vowels = 'aeiouAEIOU'
    count = sum(1 for char in name if char in vowels)
    return f"No of vowels: {count}"

def frequency_of_letters(name):
    freq = {}
    for char in name:
        freq[char] = freq.get(char, 0) + 1
    return "Frequency of letters: " + ', '.join(f"{k}-{v}" for k, v in freq.items())

#Test the module in another script
# File: main.py
import mymodule

name = input("Enter name: ")

print(mymodule.ispalindrome(name))
print(mymodule.count_the_vowels(name))
print(mymodule.frequency_of_letters(name))

'''Sample:
#Input:
bob
Output:
Yes it is a palindrome.
No of vowels: 1
Frequency of letters: b-2, o-1'''

