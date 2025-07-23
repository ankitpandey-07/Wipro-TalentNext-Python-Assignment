
#Project 1
#Q1. Write a Python function that accepts a hyphen-separated sequence of colors as input and returns the colors in a hyphen-separated sequence after sorting them alphabetically.
#A1.
def sort_colors(color_string):
    color_list = color_string.split('-')
    color_list.sort()
    return '-'.join(color_list)

#Q2. Through command line arguments three strings separated by space are given to you. Each string is a series of numbers separated by hyphen(-). You like all the numbers in string 1 and dislike all the numbers in string 2. Third string contains the numbers given to you.
#A2.
import sys

def calculate_happiness(like_str, dislike_str, data_str):
    likes = set(like_str.split('-'))
    dislikes = set(dislike_str.split('-'))
    data = data_str.split('-')
    
    happiness = 0
    for num in data:
        if num in likes:
            happiness += 1
        elif num in dislikes:
            happiness -= 1
    return happiness

# Example usage
# args = ["60-77-34-5-2", "44-11-99-3", "77-15-13-2-34-3"]
# print(calculate_happiness(*args))

#Project 2
#Q1. Write a function ispalindrome(name) that checks if the name is a palindrome.
#A1.#
def ispalindrome(name):
    return name == name[::-1]

#Q2. Write a function count_the_vowels(name) to count vowels in a name.
#A2.
def count_the_vowels(name):
    vowels = 'aeiouAEIOU'
    return sum(1 for ch in name if ch in vowels)

#Q3. Write a function frequency_of_letters(name) to count frequency of each letter.
#A3.
def frequency_of_letters(name):
    freq = {}
    for ch in name:
        freq[ch] = freq.get(ch, 0) + 1
    return freq

#Q4. Spiritual function to return the sum of all numbers in a list.
#A4.
def sum_of_list(lst):
    return sum(lst)

#Q5. Write a function to return the reverse of a string.
#A5.
def reverse_string(s):
    return s[::-1]

#Q6. Write a function to calculate the factorial of a number.
#A6.
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

#Q7. Write a function to count uppercase and lowercase letters in a string.
#A7.
def count_case(s):
    upper = sum(1 for ch in s if ch.isupper())
    lower = sum(1 for ch in s if ch.islower())
    return upper, lower

#Q8. Write a function to print even numbers from a list.
#A8.
def even_numbers(lst):
    return [x for x in lst if x % 2 == 0]

#Q9. Write a function to check if a number is prime.
#A9.
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

#Q10. Write a program to remove given items from a set.
#A10.
def remove_items(original_set, items_to_remove):
    return original_set - set(items_to_remove)

#Q11. Write a program to create an intersection of sets.
#A11.
def intersection_of_sets(set1, set2):
    return set1 & set2

#Q12. Write a program to create a union of sets.
#A12.
def union_of_sets(set1, set2):
    return set1 | set2

#Q13. Write a program to find the maximum and minimum value in a set.
#A13.
def max_min_in_set(s):
    return max(s), min(s)

