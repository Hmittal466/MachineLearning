#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


x = df.iloc[:,:-1]


# In[7]:


x


# In[8]:


y = df["label"]


# In[9]:


y


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.75,random_state=0)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)


# In[27]:


y_pred=knn.predict(x_test)


# In[28]:


y_pred


# In[29]:


y_test


# # Evaluating the Performance of model

# In[30]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


# In[31]:


accuracy_score(y_test,y_pred)


# In[32]:


precision_score(y_test,y_pred)


# In[33]:


recall_score(y_test,y_pred)


# In[34]:


f1_score(y_test,y_pred)


# In[35]:


confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
confusion_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=[False,True])
confusion_matrix.plot()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




