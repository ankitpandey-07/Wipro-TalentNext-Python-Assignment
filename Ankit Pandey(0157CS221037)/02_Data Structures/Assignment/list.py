#1. Create a list of 5 integers and display the list items. Access individual elements through index.
#Solution
numbers = [10, 20, 30, 40, 50]

print("Full list:", numbers)
print("First element:", numbers[0])
print("Second element:", numbers[1])
print("Third element:", numbers[2])
print("Fourth element:", numbers[3])
print("Fifth element:", numbers[4])

#2. Write a program to append a new item to the end of the list.
#Solution
my_list = [1, 2, 3, 4, 5]
my_list.append(6)
print("Updated list:", my_list)

#3. Write a program to reverse the order of the items in the list.
#Solution
my_list = [1, 2, 3, 4, 5]
my_list.reverse()
print("Reversed list:", my_list)

#4. Write a program to print the number of occurrences of a specified element in a list.
#Solution
my_list = [1, 2, 3, 2, 4, 2, 5]
count = my_list.count(2)
print("Number of occurrences of 2:", count)

#5. Write a program to append the items of list1 to list2 in the front.
#Solution
list1 = [4, 5, 6]
list2 = [1, 2, 3]

list2 = list1 + list2
print("New list after appending list1 in front of list2:", list2)

#6. Write a program to insert a new item before the second element in an existing list.
#Solution
my_list = [10, 20, 30, 40]
my_list.insert(1, 15)  # index 1 is before the second element
print("List after insertion:", my_list)

#7. Write a program to remove the item from a specified index in a list.
#Solution
my_list = [10, 20, 30, 40, 50]
del my_list[2]  # Removes the third element (index 2)
print("List after deleting index 2:", my_list)

#8. Write a program to remove the first occurrence of a specified element from a list.
#Solution
my_list = [5, 10, 15, 10, 20]
my_list.remove(10)  # Removes first occurrence of 10
print("List after removing first occurrence of 10:", my_list)