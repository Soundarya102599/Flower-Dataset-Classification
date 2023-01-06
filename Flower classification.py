#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv(r"C:\Users\sbabu5\Downloads\iris.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


features = df.iloc[:,:4].values
print(features)


# In[6]:


labels = df.iloc[:,4].values
print(labels)


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size = 0.20,random_state = 82)


# In[8]:


from sklearn.preprocessing import StandardScaler


# In[14]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[17]:


nvclassifier = GaussianNB()
nvclassifier.fit(X_train,y_train)


# In[18]:


preds = nvclassifier.predict(X_test)
print(preds)


# In[19]:


comp = np.vstack((y_test,preds)).T


# In[20]:


comp[:10,:]


# In[21]:


acc = accuracy_score(y_test,preds)


# In[22]:


print("Accuracy : {0:.2f}%".format(acc*100))


# In[ ]:




