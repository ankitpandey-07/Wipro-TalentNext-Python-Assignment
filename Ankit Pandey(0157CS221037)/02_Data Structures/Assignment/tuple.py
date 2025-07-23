'''1. Print the 4th element from first and 4th element from last in a tuple
t = (5, 10, 15, 20, 25, 30, 35, 40)'''
#Solution
t = (5, 10, 15, 20, 25, 30, 35, 40)
print("4th element from start:", t[3])   # Indexing starts from 0
print("4th element from end:", t[-4])

'''2. Check whether an element exists in a tuple or not
t = (10, 20, 30, 40, 50)'''
#Solution
x = int(input("Enter element to check: "))

if x in t:
    print(x, "exists in the tuple.")
else:
    print(x, "does not exist in the tuple.")

#3. Convert a list into a tuple
#Solution
lst = [1, 2, 3, 4, 5]
t = tuple(lst)

print("Converted tuple:", t)

#4. Find the index of an item in a tuple
#Solution
t = ('a', 'b', 'c', 'd', 'e')
x = input("Enter item to find index: ")

if x in t:
    print("Index of", x, "is:", t.index(x))
else:
    print(x, "is not in the tuple.")

'''5. Replace last value of tuples in a list with 100
Sample Input:
[(10, 20, 40), (40, 50, 60), (70, 80, 90)]
Expected Output:
[(10, 20, 100), (40, 50, 100), (70, 80, 100)]'''
#Solution
list_of_tuples = [(10, 20, 40), (40, 50, 60), (70, 80, 90)]

updated_list = [t[:-1] + (100,) for t in list_of_tuples]

print("Updated list:", updated_list)

