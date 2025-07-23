'''Project 1
Create a dictionary that contains a list of people and one interesting fact about each of them.
Display each person and his or her interesting fact to the screen. Next, change a fact about one of the people. Also add an additional person and corresponding fact. Display the new list of people and facts. Run the program multiple times and notice if the order changes.
Sample Output:
Jeff: is afraid of Dogs.  
David: Plays the piano.  
Jason: Can fly an airplane.  

Jeff: is afraid of heights.  
David: Plays the piano.  
Jason: Can fly an airplane.  
Jill: Can hula dance.'''
#Python Code:

facts = {
    "Jeff": "is afraid of Dogs.",
    "David": "Plays the piano.",
    "Jason": "Can fly an airplane."
}

for person, fact in facts.items():
    print(f"{person}: {fact}")

facts["Jeff"] = "is afraid of heights."
facts["Jill"] = "Can hula dance."

print("\nUpdated Facts:")
for person, fact in facts.items():
    print(f"{person}: {fact}")

'''Project 2
Given the participant's score sheet for your University Sports Day, you are required to find the runner-up score. You have scores. Store them in a list and find the score of the runner-up.
Sample Input:
[2, 3, 6, 6, 5]
Sample Output:
5
Explanation:
Given list is (2, 3, 6, 6, 5). The maximum score is 6, second maximum is 5. Hence, we print 5 as the runner-up score.'''
#Python Code:
scores = [2, 3, 6, 6, 5]
first = max(scores)
scores = [s for s in scores if s != first]
runner_up = max(scores)
print("Runner-up score:", runner_up)

'''Project 3
You have a record of n students. Each record contains the student's name, and their percent marks in Math, Physics and Chemistry. The marks can be floating values. You are required to save the record in a dictionary data type.
Student's name is the key. Marks stored in a list is the value. The user enters a student's name. Output the average percentage marks obtained by that student.
Sample Input:
"Krishna": [67, 68, 69],  
"Arjun": [70, 98, 63],  
"Malika": [52, 56, 60]
Sample Output:
Enter a name: Malika  
Average percentage mark: 56
Explanation:
Marks for Malika are [52, 56, 60] whose average is (52+56+60)/3 => 56'''
#Python Code:

records = {
    "Krishna": [67, 68, 69],
    "Arjun": [70, 98, 63],
    "Malika": [52, 56, 60]
}

name = input("Enter a name: ")
if name in records:
    marks = records[name]
    average = sum(marks) / len(marks)
    print("Average percentage mark:", int(average))
else:
    print("Student not found.")

'''Project 4
Given a string of n words, help Alex to find out how many times his name appears in the string.
Constraint:
1 <= n <= 200
Sample Input:
Hi Alex WelcomeAlex Bye Alex.
Sample Output:
3
#Explanation:
"Alex" appears 3 times in the string.'''
#Python Code:
text = "Hi Alex WelcomeAlex Bye Alex"
count = text.count("Alex")
print("Number of times 'Alex' appears:", count)

