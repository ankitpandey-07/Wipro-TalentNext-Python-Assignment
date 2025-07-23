#Q1. Write a program to accept two numbers as command line arguments and display their sum.
#Solution
import sys

num1 = int(sys.argv[1])
num2 = int(sys.argv[2])

print("Sum:", num1 + num2)

#Q2. Write a program to accept a welcome message through command line arguments and display the file name along with the welcome message.
#Solution
import sys
print("File Name:", sys.argv[0])
print("Welcome Message:", sys.argv[1])

#Q3. Write a program to accept 10 numbers through command line arguments and calculate the sum of prime numbers among them.
#Solution
import sys

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

nums = list(map(int, sys.argv[1:]))
prime_sum = sum(n for n in nums if is_prime(n))

print("Sum of prime numbers:", prime_sum)

