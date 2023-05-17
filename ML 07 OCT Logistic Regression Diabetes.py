#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


# In[7]:


x=df[features]


# In[8]:


y=df.label


# In[9]:


x


# In[10]:


y


# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,train_size = 0.75,random_state =0)


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


lr = LogisticRegression()


# In[34]:


lr.fit(x_train,y_train)


# In[35]:


y_pred = lr.predict(x_test)


# In[36]:


y_pred


# In[37]:


y_test


# # Evaluting the Performance of model

# In[38]:


from sklearn import metrics


# In[39]:


confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[False, True])
confusion_matrix.plot()
plt.show()


# In[40]:


#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[42]:


#Precision
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)


# In[43]:


#Recall
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)


# In[46]:


#F1 Score
F1_score= metrics.f1_score(y_test,y_pred)
print(F1_score)

