#Assignment
'''1. Question:
Write a program to accept two numbers from the user and perform division. If any exception occurs, print an error message or else print the result.
Explanation:
We ask the user to input two numbers. If they enter anything invalid (like a string instead of a number, or try to divide by zero), we catch the exception and print an appropriate error message.'''
#Code:
try:
    num1 = int(input("Enter first number: "))
    num2 = int(input("Enter second number: "))
    result = num1 / num2
    print("Result of division is:", result)
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
except ValueError:
    print("Error: Please enter valid integers.")
except Exception as e:
    print("Unexpected error:", e)

'''2. Question:
Write a program to accept a number from the user and check whether it's prime or not. If user enters anything other than number, handle the exception and print an error message.
Explanation:
We check if a given number is prime. If the user enters an invalid input (like a word), we handle it using try-except.'''
#Code:
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

try:
    num = int(input("Enter a number: "))
    if is_prime(num):
        print(num, "is a prime number.")
    else:
        print(num, "is not a prime number.")
except ValueError:
    print("Error: Please enter a valid integer.")

'''3. Question:
Write a program to accept the file name to be opened from the user. If file exists, print the contents of the file in title case. If not, handle the exception and print an error message.
Explanation:
We ask for the file name, try to open it, and display its contents in title case. If file doesn't exist, it throws FileNotFoundError.'''
#Code:
try:
    file_name = input("Enter the file name to open: ")
    with open(file_name, "r") as file:
        content = file.read()
        print("File content in title case:")
        print(content.title())
except FileNotFoundError:
    print("Error: The specified file does not exist.")
except Exception as e:
    print("Unexpected error:", e)



'''4. Question:
Declare a list with 10 integers and ask the user to enter an index. Check whether the number at that index is positive or negative. If any invalid index is entered, handle the exception and print an error message.
Explanation:
We use try-except to handle invalid index inputs (like out of range or non-integer).'''
#Code:
numbers = [12, -7, 5, -3, 8, -2, 0, 4, -6, 9]

try:
    index = int(input("Enter an index (0 to 9): "))
    number = numbers[index]
    if number > 0:
        print("The number at index", index, "is positive.")
    elif number < 0:
        print("The number at index", index, "is negative.")
    else:
        print("The number at index", index, "is zero.")
except IndexError:
    print("Error: Index out of range. Please enter a value between 0 and 9.")
except ValueError:
    print("Error: Please enter a valid integer index.")

