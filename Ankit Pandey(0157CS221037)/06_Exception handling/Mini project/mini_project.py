#Mini Project
'''Question:
Project 1 – Supermarket Bill Analyzer
You had saved the items and their price details in a text file, which you purchased yesterday from a nearby supermarket.
You need to know:
1.	How many items did you purchase?
2.	How many items are free?
3.	What is the total amount you had to pay?
4.	What is the discount amount?
5.	What is the final amount you paid after the discount?
Instructions:
•	Data is stored in a text file, e.g., Purchase-1.txt.
•	Each line contains one item's details. Item name and price are separated by a space.
•	Free items will have “free” or “Free” as their price.
•	A blank line indicates separation between purchased and free items or between items and the discount.
•	The line with the word Discount followed by a number indicates the discount amount.
•	You must enter the file name during program execution.
•	Handle possible exceptions such as file not found or incorrect formatting.
________________________________________
 Explanation:
The program:
•	Takes a file name input from the user.
•	Reads the file and categorizes lines into:
o	Paid items with their price.
o	Free items marked as Free.
o	Discount line, found after a blank line, marked with Discount <number>.
•	Calculates:
o	Total number of paid items.
o	Total number of free items.
o	Total price of all paid items.
o	Final price after subtracting the discount.
•	Also, handles any invalid or missing file gracefully with try-except.'''

 #Python Code:
def analyze_bill():
    try:
        filename = input("Enter the file name (with .txt): ")
        with open(filename, "r") as file:
            lines = [line.strip() for line in file if line.strip() != ""]

        paid_items = 0
        free_items = 0
        total_amount = 0
        discount = 0
        discount_found = False

        for line in lines:
            if line.lower().startswith("discount"):
                # Extract discount
                try:
                    discount = int(line.split()[1])
                    discount_found = True
                except:
                    print("Invalid discount line.")
                    return
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                price_str = parts[-1]
                if price_str.lower() == "free":
                    free_items += 1
                else:
                    try:
                        price = int(price_str)
                        total_amount += price
                        paid_items += 1
                    except ValueError:
                        print(f"Invalid price format in line: {line}")
                        return

        final_amount = total_amount - discount

        print(f"\nNo of items purchased: {paid_items}")
        print(f"No of free items: {free_items}")
        print(f"Amount to pay: {total_amount}")
        print(f"Discount given: {discount}")
        print(f"Final amount paid: {final_amount}")

    except FileNotFoundError:
        print("File not found. Please make sure the file name is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
analyze_bill()
'''
 Sample Output 1:
Input (Purchase-1.txt):
Chocolate 50
Biscuit 35
Icecream 50

Discount 5
Console Output:
Enter the file name (with .txt): Purchase-1.txt

No of items purchased: 3
No of free items: 0
Amount to pay: 135
Discount given: 5
Final amount paid: 130'''

'''Sample Output 2:
Input (Purchase-2.txt):
Chocolate 50
Biscuit 35
Icecream 50
Rice 100
Chicken 250

Perfume Free
Soup Free

Discount 80
Console Output:
Enter the file name (with .txt): Purchase-2.txt

No of items purchased: 5
No of free items: 2
Amount to pay: 485
Discount given: 80
Final amount paid: 405
'''
