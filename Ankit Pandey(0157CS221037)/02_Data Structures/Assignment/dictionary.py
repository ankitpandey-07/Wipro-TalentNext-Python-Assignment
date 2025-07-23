'''1. Write a program to add a key and value to a dictionary.
Sample Dictionary: {0: 10, 1: 20}
Expected Result: {0: 10, 1: 20, 2: 30}'''
#Solution
d = {0: 10, 1: 20}
d[2] = 30
print("Updated Dictionary:", d)

'''2. Write a program to concatenate the following dictionaries to create a new one.
Sample Dictionaries:
dic1 = {1: 10, 2: 20}  
dic2 = {3: 30, 4: 40}  
dic3 = {5: 50, 6: 60}
Expected Result: {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}
dic1 = {1: 10, 2: 20}
dic2 = {3: 30, 4: 40}
dic3 = {5: 50, 6: 60}'''
#Solution
dic1 = {1: 10, 2: 20}
dic2 = {3: 30, 4: 40}
dic3 = {5: 50, 6: 60}
merged_dict = {}
for d in (dic1, dic2, dic3):
    merged_dict.update(d)

print("Concatenated Dictionary:", merged_dict)

#3. Write a program to check if a given key already exists in a dictionary.
#Solution
d = {1: 100, 2: 200, 3: 300}
key = int(input("Enter a key to check: "))

if key in d:
    print("Key exists in the dictionary.")
else:
    print("Key does not exist.")

'''4. Write a program to iterate over a dictionary using for loop and print:
•	Keys alone
•	Values alone
•	Both keys and values
d = {1: "Apple", 2: "Banana", 3: "Cherry"}'''
#Solution
print("Keys:")
for key in d:
    print(key)

print("\nValues:")
for value in d.values():
    print(value)

print("\nKey-Value Pairs:")
for key, value in d.items():
    print(f"{key}: {value}")

#5. Write a program to prepare a dictionary where the keys are numbers between 1 and 15 (both included) and the values are squares of the keys.
#Solution
squares = {}
for i in range(1, 16):
    squares[i] = i ** 2

print("Dictionary of squares:", squares)

#6. Write a program to sum all the values in a dictionary.
#Solution
d = {'a': 100, 'b': 200, 'c': 300}
total = sum(d.values())

print("Sum of all values:", total)

