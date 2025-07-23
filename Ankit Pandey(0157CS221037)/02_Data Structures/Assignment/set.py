#1. Remove a given item from a set
#Solution
my_set = {10, 20, 30, 40, 50}
item = int(input("Enter item to remove: "))

my_set.discard(item)    # discard doesn't raise error if item not found

print("Updated set:", my_set)

#2. Create an Intersection of sets
#Solution
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

intersection = set1 & set2    # or set1.intersection(set2)

print("Intersection:", intersection)

#3. Create a Union of sets
#Solution
set1 = {1, 2, 3}
set2 = {3, 4, 5}

union = set1 | set2  # or set1.union(set2)

print("Union:", union)

#4. Find the Maximum and Minimum value in a set
#Solution
my_set = {12, 45, 3, 67, 29}

print("Maximum value:", max(my_set))
print("Minimum value:", min(my_set))

