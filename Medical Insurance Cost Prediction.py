#!/usr/bin/env python
# coding: utf-8

# In[61]:


#Import the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[63]:


#data collection and  exploratory data analysis
Insurance_dataset = pd.read_csv("insurance.csv")
Insurance_dataset.head()


# In[65]:


Insurance_dataset.shape


# In[67]:


Insurance_dataset.info()


# In[69]:


Insurance_dataset.describe()


# In[71]:


Insurance_dataset.isnull().sum()


# In[73]:


# Before analysis in depth, data is usually seperated into categorical data and numerical data
# this is because the categorical data cannot be used to train models, but numbers are the key factors in the trainining


# In[75]:


#Analyse age distribution
sns.set()
plt.figure(figsize = (6,6))
sns.histplot(Insurance_dataset['age']) #similarly, you can use distplot in case of histplot
plt.title('Age Distribution')
plt.show()


# In[77]:


#gender distribution
plt.figure(figsize = (6,6))
sns.countplot(x ='sex', data = Insurance_dataset)
plt.title('Gender Distribution')
plt.show()


# In[79]:


#alternatively we can do this
Insurance_dataset['sex'].value_counts()


# In[81]:


#Body mass index distribution
sns.set()
plt.figure(figsize = (6,6))
sns.histplot(Insurance_dataset['bmi']) #similarly, you can use distplot instead of histplot
plt.title('BMI Distribution')
plt.show()


# In[83]:


#normal BMI between 18.5 and 24.9


# In[85]:


#Distribution of Children Columns
plt.figure(figsize = (6,6))
sns.countplot(x ='children', data = Insurance_dataset)
plt.title('Children')
plt.show()


# In[87]:


#alternatively we can do this
Insurance_dataset['children'].value_counts()


# In[89]:


#Distribution of smokers
plt.figure(figsize = (6,6))
sns.countplot(x ='smoker', data = Insurance_dataset)
plt.title('Smoker Distribution')
plt.show()


# In[91]:


plt.figure(figsize = (6,6))
sns.countplot(x ='region', data = Insurance_dataset)
plt.title('Distribution of Regions')
plt.show()


# In[93]:


#insurance charge distribution
plt.figure(figsize = (6,6))
sns.histplot(Insurance_dataset['charges']) #similarly, you can use distplot instead of histplot
plt.title('Charge Distribution')
plt.show()


# In[ ]:





# In[96]:


#Data pre-processing
#we have 3 columns with categorical data but the computer only understands numerical data so w e will either drop those columns, or change the categorical data to numerical.


# In[104]:


#encoding categorical features
Insurance_dataset.replace({'sex':{'male': 0, 'female': 1}}, inplace = True) #encoding sex column
Insurance_dataset.replace({'smoker':{'no': 0, 'yes': 1}}, inplace = True) #encoding smoker column
Insurance_dataset.replace({'region':{'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace = True) #encoding region column
#Insurance_dataset = Insurance_dataset.infer_objects(copy=False)# Explicitly infer object types without copying
Insurance_dataset.head()


# In[106]:


#Splitting the feature and target
#we split the features (input variables) and target (output variable) to clearly define what the model will learn from and what it will predict.


# In[110]:


X = Insurance_dataset.drop(columns = 'charges' , axis = 1)
Y = Insurance_dataset['charges'] #y=f(x)
print(X)


# In[ ]:





# In[113]:


print(Y)


# In[123]:


#Splitting the data into training and testing set .
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)


# In[127]:


#Model training
regressormodel = LinearRegression()
#fit model
regressormodel.fit(X_train, Y_train)


# In[133]:


#Evaluate model
y_train_pred = regressormodel.predict(X_train)
y_train_pred


# In[137]:


#R squared values
r2_train = metrics.r2_score(Y_train, y_train_pred)
print('R2 value: ', r2_train)#value ususally between 0 and 1


# In[141]:


#prediction on testing set
y_test_pred = regressormodel.predict(X_test)
y_test_pred


# In[143]:


r2_test = metrics.r2_score(Y_test, y_test_pred)
print('R2 value: ', r2_test)#value ususally between 0 and 1


# In[149]:


#Building a predictive system
Input_data = (31, 1, 25.74, 0, 0, 0) #tuples in python
#input data as an array
input_data_as_array = np.asarray(Input_data)
#reshape data
input_data_reshape = input_data_as_array.reshape(1, -1)
#make prediction
prediction = regressormodel.predict(input_data_reshape)
#print prediction
print(prediction)
print('The insurance cost in USD is: ', prediction[0])


# In[ ]:




