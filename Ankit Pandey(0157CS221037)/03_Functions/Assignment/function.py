#1. Function to return the sum of all numbers in a list

def sum_of_numbers(lst):
    total = 0
    for item in lst:
        if isinstance(item, int) or isinstance(item, float):
            total += item
    return total

# Example:
print(sum_of_numbers([1, 2, 3, 'P', 7]))  # Output: 13

#2. Function to return the reverse of a string.
#Solution
def reverse_string(s):
    return s[::-1]

# Example:
print(reverse_string("1234abcd"))  # Output: "dcba4321"

#3. Function to calculate and return the factorial of a number.
#Solution
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Example:
print(factorial(5))  # Output: 120

#4. Function to count uppercase and lowercase letters in a string.
#Solution
def count_case(s):
    upper = 0
    lower = 0
    for char in s:
        if char.isupper():
            upper += 1
        elif char.islower():
            lower += 1
    print(f"Upper case letters: {upper}")
    print(f"Lower case letters: {lower}")

# Example:
count_case("HelloWorld")  # Output: Upper: 2, Lower: 8

#5. Function to print even numbers from a given list
#Solution
def print_even_numbers(lst):
    evens = [num for num in lst if num % 2 == 0]
    print(evens)

# Example:
print_even_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Output: [2, 4, 6, 8]

#6. Function to check if a number is prime
#Solution
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

# Example:
print(is_prime(7))   # Output: True
print(is_prime(10))  # Output: False

