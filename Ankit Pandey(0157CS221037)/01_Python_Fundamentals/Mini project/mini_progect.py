"""
Project: 1
Create a python program that asks the user how far they want to travel. If they want to travel less than three miles tell them to ride Bicycle. If they want to travel more than three miles, but less than three hundred miles, tell them to ride Motor-Cycle. If they want to travel three hundred miles or more tell them to drive Super-Car.

Sample Output:

How far would you like to travel in miles? 2500

I suggest Super-Car to your destination
"""
#Python Code:

distance = int(input("How far would you like to travel in miles? "))

if distance < 3:
    print("I suggest Bicycle to your destination")
elif distance < 300:
    print("I suggest Motor-Cycle to your destination")
else:
    print("I suggest Super-Car to your destination")

"""
Project: 2
Let's assume you are planning to use your Python skills to build a PBLApp for Mobile.

You decide to host your application on servers running in the cloud. You pick a hosting provider that charges $0.51 per hour. You will launch your service using one server and want to know how much it will cost to operate per day, per week, per month.

Write a Python program that displays the answers to the following questions:

How much does it cost to operate one server per day?
How much does it cost to operate one server per week?
How much does it cost to operate one server per month?
How many days can I operate one server with $918?
"""
#Python Code:

hourly_rate = 0.51

daily_cost = hourly_rate * 24
weekly_cost = daily_cost * 7
monthly_cost = daily_cost * 30
budget = 918
days_with_budget = budget / daily_cost

print("Cost per day: $", round(daily_cost, 2))
print("Cost per week: $", round(weekly_cost, 2))
print("Cost per month: $", round(monthly_cost, 2))
print("Number of days server can run with $918: ", int(days_with_budget))
