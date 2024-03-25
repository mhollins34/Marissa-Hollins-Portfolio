#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[3]:


## Read csv file using Pandas

Titanic_Data = pd.read_csv('titanic_data.csv')
Titanic_Data


# In[56]:


Titanic_Data.head()


# In[4]:


## Converted the Sex column into binary classes and converted the data type to integer

gender = pd.get_dummies(Titanic_Data['Sex'], drop_first=True).astype('int64')
gender


# In[5]:


## Added the encoded 'Sex' column into the dataframe with the label 'Gender'

Titanic_Data['Gender']=gender
Titanic_Data


# In[6]:


## Dropped columns that were irrelevant to regression model

Titanic_Data.drop(['Name','Sex','Ticket','Fare','Cabin','Embarked','PassengerId','SibSp','Parch'], axis=1, inplace=True)


# In[60]:


Titanic_Data


# In[61]:


Titanic_Data.dtypes


# In[62]:


Titanic_Data['Age'].mean()


# In[63]:


## Converted the Nan in the 'Age' column to numeric data using Numpy

Titanic_Data .replace('Nan', np.nan)


# In[7]:


## Filled the null values in the 'Age' column with rounded mean value of 'Age' 

Titanic_Data_Final=Titanic_Data.fillna(30)
Titanic_Data_Final


# In[65]:


Titanic_Data_Final.isnull().sum()


# In[66]:


Titanic_Data_Final.head()


# In[8]:


## Generated categorical scatter plot to reveal patterns in passenger classes based on 'Age' and 'Gender'
 
sns.catplot(x='Gender', y='Age', data=Titanic_Data_Final, hue='Pclass', aspect=2)
plt.xlabel('Gender', fontsize=20)
plt.ylabel('Age', fontsize=20)
plt.show()


# In[9]:


## Used correlation fuction to see relation between different data fields

Titanic_Data_Final.corr()


# In[10]:


## Generated heatmap of data correlation using Seaborn

sns.heatmap(Titanic_Data_Final.corr(), cmap='YlOrBr')
plt.show()


# In[83]:


## Produced histogram to reveal quantities of different age groups aboard Titanic

Titanic_Data_Final.plot.hist(y='Age')


# In[84]:


## Histogram plot showing the quantity of each ticket class held

Titanic_Data_Final.plot.hist(y='Pclass')


# In[70]:


## Assigned 'Survived' column to target variable Y

Y = Titanic_Data_Final['Survived']
Y


# In[71]:


## Dropped the 'Survived' column to assign the remaining columns to predictor variable X

X = Titanic_Data_Final.drop(['Survived'], axis=1)
X


# In[72]:


## Split the X and Y variables into train and test data

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=5)


# In[73]:


X_train


# In[74]:


Y_train


# In[75]:


## Fit training data to Logistic Regression model

LR=LogisticRegression()
LR.fit(X_train,Y_train)


# In[76]:


## Generated target predictions using the test data in the model. Passed predicte values into Pandas dataframe

Predict = LR.predict(X_test)
Predict

df = pd.DataFrame(LR.predict(X_test),columns=['Survived'])
df


# In[77]:


df.value_counts()


# In[78]:


## Generated confusion matrix to measure accuracy of logistic Regression model
confusion_matrix(Y_test, Predict)


# In[79]:


pd.DataFrame(confusion_matrix(Y_test, Predict), columns=['Predicted No','Predicted Yes'], index=['Actual No','Actual Yes'])


# In[ ]:




