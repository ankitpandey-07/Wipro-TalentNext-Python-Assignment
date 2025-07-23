#Project 1: 
'''Secret Message Decoder from a Text File
Question
Your friend has sent you a text file containing n lines. He sent a secret message with it, which tells you the place and time where you have to go and meet him.
He challenges you to find it without seeing the content of the file directly. Instead, you must write a Python program that analyses the text file and extracts the hidden information using the following hints:
________________________________________
Hints to find the secret message:
1.	The number of lines in the file tells you the meeting time.
o	Condition: 1 × number of lines < 24
o	If the number of lines is greater than 12, convert it to 12-hour format.
	E.g., If number of lines = 15 → Time is 3 PM
	If number of lines = 10 → Time is 10 AM
2.	The word appearing the maximum number of times tells you the meeting place.
o	The meeting place will be a street name formed by the most frequent word + " Street".
	E.g., If the word Park appears the most → Meeting place: Park Street
________________________________________
 Explanation of the Sample
Let's say your text file (sample.txt) contains 20 lines and the word Apollo appears 25 times, which is the highest frequency of any word.
Then:
•	20 lines → 8 PM (12-hour format of 20)
•	Most frequent word: Apollo → Place: Apollo Street'''

#Python Code
from collections import Counter

def get_meeting_details(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Step 1: Calculate meeting time
        num_lines = len(lines)
        if num_lines > 12:
            meeting_time = f"{num_lines % 12} PM"
        else:
            meeting_time = f"{num_lines} AM"

        # Step 2: Find the most frequent word
        all_words = []
        for line in lines:
            words = line.replace(",", "").replace(".", "").split()
            all_words.extend(words)

        # Count the frequency of each word (case-sensitive as per example)
        word_counts = Counter(all_words)
        most_common_word, _ = word_counts.most_common(1)[0]

        # Step 3: Prepare output
        meeting_place = f"{most_common_word} Street"

        print(f"Meeting time: {meeting_time}")
        print(f"Meeting place: {meeting_place}")

    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
get_meeting_details("sample.txt")

'''Output (Based on Sample File 1)
Meeting time: 9 AM
Meeting place: Park Street
________________________________________
 Output (Based on Sample File 2)
Meeting time: 8 PM
Meeting place: Apollo Street'''

