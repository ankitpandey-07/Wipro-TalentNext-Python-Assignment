# Q1 - Write a LC (List Comprehension) program to create an output dictionary which contains only the odd numbers that are present in the input list:[1, 2, 3, 4, 5, 6, 7] as keys, and their cubes as values.

input_list = [1, 2, 3, 4, 5, 6, 7]
output_dict = {x: x**3 for x in input_list if x % 2 != 0}
print(output_dict)


# Q2 - Make a dictionary of the 26 English alphabets mapping each with the corresponding integer.

import string
alphabet_dict = {char: idx+1 for idx, char in enumerate(string.ascii_lowercase)}
print(alphabet_dict)