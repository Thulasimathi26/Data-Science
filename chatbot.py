#!/usr/bin/env python
# coding: utf-8

# # Variables and Data types

# In[1]:


# String variable
name = "Alice"

# Integer variable (whole number)
age = 30

# Float variable (decimal number)
pi = 3.14

# Boolean variable (True or False)
is_happy = True

# Printing variable values and their data types
print("Name:", name, "is of type", type(name))
print("Age:", age, "is of type", type(age))
print("Pi:", pi, "is of type", type(pi))
print("Is happy:", is_happy, "is of type", type(is_happy))


# # Indentation

# In[4]:


age = 10

if age >= 13:
 print("You are old enough to ride the rollercoaster!")
else:
  print("Sorry, you are not tall enough yet.")


# # String Operations

# In[5]:


# Concatenation
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(full_name)  


# In[6]:


# Indexing
message = "Hello, world!"
print(message[9]) 


# In[7]:


# Slicing
greeting = "Hello, world!"
print(greeting[7:12]) 


# In[8]:



# len()
message = "Hello, world!"
print(len(message)) 


# In[9]:


# upper()
message = "hello, world!"
print(message.upper()) 


# In[10]:



# capitalize()
message = "hello, world!"
print(message.capitalize())


# In[11]:


# find()
text = "Hello, welcome to Python programming!"
result = text.find("Python")
print(result)


# In[12]:


# replace()
message = "Hello, world!"
new_message = message.replace("world", "there")
print(new_message)


# In[13]:



# split()
message = "Hello, world!"
words = message.split()
print(words) 


# # Arithmetic, Comparison and Logical Operators

# In[14]:


# Get two numbers from the user
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))

# Arithmetic operations
sum = num1 + num2
difference = num1 - num2
product = num1 * num2
division = num1 / num2

floor_division = num1 // num2

remainder = num1 % num2

# Comparison operations
is_equal = num1 == num2
is_greater = num1 > num2
is_less = num1 < num2

# Logical operations (using the results from calculations)
is_greater_and_even = is_greater and (num2 % 2 == 0)  # Check if greater and even
is_less_or_equal = is_less or (num1 == num2)  # Check if less or equal

# Print the results
print("Sum:", sum)
print("Difference:", difference)
print("Product:", product)
print("Division:", division)
print("Floor Division:", floor_division)
print("Remainder:", remainder)

print("Numbers are equal:", is_equal)
print("First number is greater:", is_greater)
print("First number is less:", is_less)

print("First number is greater and second is even (AND):", is_greater_and_even)
print("First number is less or they are equal (OR):", is_less_or_equal)


# # Conditional Statement

# In[16]:


# Get age from the user
age = int(input("Enter your age: "))

# Check age eligibility for voting
if age >= 18:
    print("You are eligible to vote.")
elif 13 <= age < 18:  # elif for checking between ranges
    print("You are not eligible to vote yet, but you can register to pre-vote soon!")
else:
    print("You are not eligible to vote at this age.")


# # Loops

# In[18]:


# Print even numbers from 2 to 10 (using for loop with condition)
for num in range(2, 11, 2):  # Step of 2 to get even numbers
    print(num)


# In[19]:


# Count from 1 to 5 and print the numbers

count = 1
while count <= 5:
  print(count)
  count += 1  # Increment count by 1 after each iteration


# # Break Statement

# In[20]:


numbers = [1, 4, 6, 5, 3, 9, 2, 8]

for number in numbers:
  if number == 5:
    print("Number found. Exiting loop.")
    break  # Exit the loop if number is 5
  print(number)

print("Loop finished.")


# # Continue Statement

# In[21]:


numbers = [1, 4, 6, 3, 9, 2, 8]

for number in numbers:
  if number % 2 == 0:  # Check if number is even
    continue  # Skip even numbers
  print(number)

print("Loop finished.")


# # Lists

# In[22]:



fruits = ["apple", "banana", "cherry"]
first_fruit = fruits[0]  
fruits.append("orange")
print(fruits)  


# # Tuples

# In[24]:


countries = ("France", "Italy", "Germany")
#countries[0] = "Spain"  
print(countries)


# # Dictionaries

# In[25]:


student = {
    "name": "Alice",
    "age": 20,
    "course": "Computer Science"
}

student_name = student["name"]  

student["major"] = "Software Engineering"

print(student)


# # Functions

# In[26]:


def greet(name):
  print(f"Hello, {name}!")
greet("Alice")


# # Modules and Packages

# In[27]:


import math

result = math.sqrt(25)
print(result)  


# # NumPy

# In[ ]:


import numpy as np
array = np.array([1, 2, 3, 4, 5])
doubled_array = array * 2
print(doubled_array)


# # Pandas

# In[28]:


import pandas as pd
data = {"Name": ["Alice", "Bob", "Charlie"], "Age": [20, 25, 22]}
df = pd.DataFrame(data)
print(df)


# # Matplotlib

# In[29]:


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sample Line Plot")
plt.show()


# In[4]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the updated dataset
df = pd.read_csv('dataset.csv')

# Preprocess and vectorize the questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Question'])

# Function to find the closest matching question
def find_answer(user_input):
    # Transform the user input to match the vectorized questions
    user_input_vect = vectorizer.transform([user_input])
    
    # Compute the cosine similarity between user input and questions
    similarities = cosine_similarity(user_input_vect, X)
    
    # Get the index of the most similar question
    best_match_idx = similarities.argmax()
    
    # Return the corresponding answer
    return df['Answer'].iloc[best_match_idx]

# Chatbot conversation loop
def chatbot():
    print("Chatbot: Hi! How can I help you today?")
    
    while True:
        user_input = input("You: ").lower()
        
        if 'bye' in user_input:
            print("Chatbot: Goodbye! Have a nice day!")
            break
        
        # Find and print the best answer
        answer = find_answer(user_input)
        print(f"Chatbot: {answer}")

# Run the chatbot
chatbot()


# In[ ]:





# In[ ]:




