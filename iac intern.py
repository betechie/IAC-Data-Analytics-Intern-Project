#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
df=pd.read_excel(r"C:\Users\Saikiran\Downloads\Data analyst Data (1).xlsx")


# In[58]:


df.columns


# In[59]:


#Missing Value Detection
df.isna().any()


# In[60]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Assuming you have your DataFrame named 'df'

# Define the columns with missing values
columns_with_missing = ['College Name', 'How did you come to know about this event?',
                        'Specify in "Others" (how did you come to know about this event)']

# Create a SimpleImputer object with the strategy as 'most_frequent' to fill with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')

# Fit and transform the imputer on the specified columns
df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])


# In[61]:


#Missing Value Detection
df.isna().any()


# In[62]:


#unique students in each column
unique_students = df['Email ID'].nunique()
unique_students


# In[63]:


#Dimensionality Check
df.shape


# In[64]:


#type of Dataset
type(df)


# In[65]:


df.mean()


# In[66]:


df.median()


# In[67]:


df.mode(axis=0)


# In[68]:


import seaborn as sns
sns.boxplot(x=df['CGPA'])


# In[69]:


df.shape


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns
correlations = df.corr()
sns.heatmap(data = correlations,square = True, cmap = "bwr")
plt.yticks(rotation=0)
plt.xticks(rotation=90)


# In[71]:


#average gpa 
average_gpa = df['CGPA'].mean()
average_gpa


# In[72]:


# the distribution of the students across different graduation years
import seaborn as sns
import matplotlib.pyplot as plt
# Create a count plot for the distribution of students across graduation years
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year of Graduation')
plt.title('Distribution of Students Across Graduation Years')
plt.xlabel('Year of Graduation')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.show()


# In[73]:


#distribution of students experience with python programing
# Create a histogram for the distribution of students' experience with Python programming
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Experience with python (Months)', bins=20)
plt.title('Distribution of Students\' Experience with Python Programming')
plt.xlabel('Months of Experience')
plt.ylabel('Number of Students')
plt.show()


# In[76]:


#avg family income of the student
# Define a function to convert income ranges to numerical values
def convert_income_range(income_range):
    if income_range == '7 Lakh+':
        return 700000
    elif income_range == '0-2 Lakh':
        return 100000
    elif income_range == '5-7 Lakh':
        return 600000
    elif income_range == '2-5 Lakh':
        return 350000
    else:
        return 0  # Handle other cases as needed

# Given dataset of Family Income values
income_values = [
    '7 Lakh+',
    '0-2 Lakh',
    '5-7 Lakh',
    '2-5 Lakh',
    '0-2 Lakh',
    '0-2 Lakh',
    '2-5 Lakh',
    '0-2 Lakh',
    '0-2 Lakh',
    '0-2 Lakh',
    '0-2 Lakh',
]

# Convert income ranges to numerical values and calculate the sum
total_numerical_values = sum(convert_income_range(income_range) for income_range in income_values)

# Calculate the average
number_of_values = len(income_values)
average = total_numerical_values / number_of_values

# Print the average
print(f"The average Family Income is: {average:.2f}")


# In[77]:


#gpa vary among different colleges (show top 5 results only)
# Create a bar plot for GPA variation among different colleges (top 5 results)
top_colleges = df.groupby('College Name')['CGPA'].mean().nlargest(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_colleges.index, y=top_colleges.values)
plt.title('Top 5 Colleges: GPA Variation')
plt.xlabel('College Name')
plt.ylabel('Average GPA')
plt.xticks(rotation=45)
plt.show()


# In[78]:


#are there any outliers in the quantity (number of courses completed) attribute
sns.boxplot(x=df['Quantity'])


# In[80]:


# Calculate the average GPA for students from each city
average_gpa_by_city = df.groupby('City')['CGPA'].mean()

# Display the average GPA for each city
average_gpa_by_city


# In[81]:



# Define a function to convert income ranges to numerical values
def convert_income_range(income_range):
    if income_range == '7 Lakh+':
        return 700000
    elif income_range == '0-2 Lakh':
        return 100000
    elif income_range == '5-7 Lakh':
        return 600000
    elif income_range == '2-5 Lakh':
        return 350000
    else:
        return 0  

# Convert income ranges to numerical values in the DataFrame
df['Family Income'] = df['Family Income'].apply(convert_income_range)

# Create a scatter plot between family income and GPA
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Family Income', y='CGPA')
plt.title('Scatter Plot: Family Income vs GPA')
plt.xlabel('Family Income')
plt.ylabel('GPA')
plt.show()

# Calculate the correlation coefficient between family income and GPA
correlation = df['Family Income'].corr(df['CGPA'])

# Display the correlation coefficient
correlation


# In[82]:


# Create a count plot to visualize the number of students from each city
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='City')
plt.title('Number of Students from Various Cities')
plt.xlabel('City')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.show()


# In[83]:



# Create scatter plots or regression plots to visualize the relationships
plt.figure(figsize=(16, 6))

# Scatter plot: Expected Salary vs GPA
plt.subplot(1, 3, 1)
sns.scatterplot(data=df, x='CGPA', y='Expected salary (Lac)')
plt.title('Expected Salary vs GPA')
plt.xlabel('GPA')
plt.ylabel('Expected Salary')

# Scatter plot: Expected Salary vs Family Income
plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Family Income', y='Expected salary (Lac)')
plt.title('Expected Salary vs Family Income')
plt.xlabel('Family Income')
plt.ylabel('Expected Salary')

# Scatter plot: Expected Salary vs Experience with Python
plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Experience with python (Months)', y='Expected salary (Lac)')
plt.title('Expected Salary vs Experience with Python')
plt.xlabel('Experience with Python (Months)')
plt.ylabel('Expected Salary')

plt.tight_layout()
plt.show()


# In[94]:


# Create a count plot to visualize the distribution of attendees from different fields of study for each event
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Events', hue='Designation')  # Replace 'Designation' with the correct column name
plt.title('Distribution of Students\' Fields of Study for Each Event')
plt.xlabel('Event')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.legend(title='Field of Study')
plt.tight_layout()
plt.show()


# In[87]:


#Do students in leadership positions tend to have higher GPAs?
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Leadership- skills', y='CGPA')
plt.title('Leadership Skills vs GPA')
plt.xlabel('Leadership Skills')
plt.ylabel('GPA')
plt.show()


# In[88]:


#. How many students graduating by the end of 2024 tend to have higher GPAs?
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year of Graduation', hue='CGPA')
plt.title('Distribution of GPAs for Different Graduation Years')
plt.xlabel('Year of Graduation')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[89]:


#Which promotion channel brings in more student participations for the event?
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='How did you come to know about this event?', hue='Attendee Status')
plt.title('Distribution of Students\' Participation by Promotion Channel')
plt.xlabel('Promotion Channel')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[90]:


#Find the total number of students who attended events related to Data Science?
data_science_events = ['Data Science Event 1', 'Data Science Event 2', 'Data Science Event 3']
data_science_attendees = df[df['Events'].isin(data_science_events)].shape[0]
print("Total number of students who attended Data Science events:", data_science_attendees)


# In[91]:


#Do students with high CGPA and more experience tend to have higher salary expectations?
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='CGPA', y='Expected salary (Lac)', hue='Experience with python (Months)')
plt.title('Expected Salary vs CGPA with Experience')
plt.xlabel('CGPA')
plt.ylabel('Expected Salary')
plt.legend(title='Experience with Python')
plt.show()


# In[93]:


# How many students know about the event from their colleges? Which of these Top 5 colleges?

top_colleges = df['College Name'].value_counts().nlargest(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_colleges.index, y=top_colleges.values)
plt.title('Top 5 Colleges: Students\' Awareness about the Event from Their Colleges')
plt.xlabel('College Name')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




