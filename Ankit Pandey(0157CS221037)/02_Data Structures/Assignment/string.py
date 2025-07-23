#1. Count the number of upper and lower case letters in a string
#Solution
text = input("Enter a string: ")
upper = 0
lower = 0

for char in text:
    if char.isupper():
        upper += 1
    elif char.islower():
        lower += 1

print("Uppercase letters:", upper)
print("Lowercase letters:", lower)

#2. Check whether a given string is palindrome or not
#Solution
s = input("Enter a string: ")

if s == s[::-1]:
    print("Palindrome")
else:
    print("Not a palindrome")

#3. Return a new string made of n copies of the first 2 characters, where n is the length of the string
#Solution
s = input("Enter a string: ")

if len(s) >= 2:
    front = s[:2]
    result = front * len(s)
    print(result)
else:
    print("String length must be greater than 2")

#4. Remove 'x' if it appears as first or last character in the string
#Solution
s = input("Enter a string: ")

if s.startswith('x'):
    s = s[1:]
if s.endswith('x'):
    s = s[:-1]

print("Updated string:", s)

#5. Return a string made of n repetitions of the last n characters of the string
#Solution
s = input("Enter a string: ")
n = int(input("Enter a number: "))

if 0 <= n <= len(s):
    result = s[-n:] * n
    print(result)
else:
    print("Invalid input for n")

