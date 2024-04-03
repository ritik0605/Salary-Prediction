#!/usr/bin/env python
# coding: utf-8

# # Title: 
# Salary Prediction using Machine Learning

# # Objective: 
# To develop a machine learning model that predicts salary based on years of experience.

# # Problem Statement: 
# Build a regression model to accurately predict salaries given the number of years of experience. This model can be used by organizations for salary negotiations, workforce planning, and budgeting purposes.

# # Tools: 
# Python, Scikit-learn, Pandas, Matplotlib

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # Load the dataset

# In[2]:


df=pd.read_csv(r"C:\Users\Ritik Sonwane\OneDrive\Desktop\Salary_Data.csv")
df


# # Data Preprocessing

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.columns


# # Relationship between the salary and job experience of the people

# In[9]:


figure=px.scatter(data_frame=df,x="Salary",y="YearsExperience",size="YearsExperience", trendline="ols")
figure.show()


# # Splitting the Data

# In[11]:


x=np.asanyarray(df[["YearsExperience"]])
y=np.asanyarray(df[["Salary"]])
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# # Model Selection and Training

# In[12]:


model = LinearRegression()
model.fit(xtrain,ytrain)


# # Model Evaluation

# In[13]:


train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)

print(f"Training Score: {train_score}")
print(f"Test Score: {test_score}")


# # Salary Prediction

# In[14]:


a = int(input("Years of Experience : "))
features = np.array([[a]])
print(f"Predicted Salary for {a} years of experience is:",model.predict(features))


# In[ ]:




