#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


covid = pd.read_csv("covidreports.csv")


# In[3]:


covid.head()


# In[4]:


covid.shape


# In[5]:


covid.columns


# In[6]:


covid['Infected'].value_counts()


# In[7]:


covid.isna().any()


# In[8]:


covid.info()


# In[9]:


covid.describe()


# In[10]:


covid.groupby('Country')['Infected'].sum()


# In[11]:


covid.groupby('Country')['Infected','fever'].sum()


# In[12]:


covid.groupby('Country')['Infected','Bodypain'].sum()


# In[13]:


covid.groupby('Country')['Infected','Runny_nose'].sum()


# In[14]:


covid.groupby('Country')['Infected','Difficulty_in_breathing'].sum()


# In[15]:


covid.groupby('Country')['Infected','Nasal_congestion','Sore_throat'].sum()


# In[16]:


covid.hist(bins=20, figsize=(15,10))


# In[17]:


corr_matrix = covid.corr()
corr_matrix["Infected"].sort_values(ascending=False)


# In[18]:


plt.figure(figsize= (20, 20))
sns.heatmap(covid.corr(), annot = True )


# In[19]:


plt.figure(figsize=(12,6))
sns.countplot(data=covid, x="Infected" , hue='Severity')


# In[20]:


India_case=covid[covid["Country"]=="India"]
India_case.head()


# In[21]:


x=['Age']
y=['Infected']


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test=train_test_split(x,y, test_size=0.3,random_state=30)


# In[48]:


x_train.shape


# In[49]:


y_train.shape


# In[50]:


x_test.shape


# In[51]:


y_test.shape


# # Build model

# # Using Random Forest

# # train

# In[52]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[53]:


rnd_clf=RandomForestClassifier()
rnd_clf.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))


# In[54]:


rnd_clf.score(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))
y_rnd_pred=rnd_clf.predict(np.array(x_test).reshape(-1,1))


# In[55]:


y_rnd_pred= rnd_clf.predict(np.array(x_train).reshape(-1,1))


# In[56]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy after CV :", accuracy_score(y_train, y_rnd_pred))
print("Pricision after CV:", precision_score(y_train, y_rnd_pred))
print("Recall after CV   :", recall_score(y_train, y_rnd_pred))
print("f1_score after CV :", f1_score(y_train, y_rnd_pred))


# # test

# In[57]:


y_test_rnd_pred= rnd_clf.predict(np.array(x_test).reshape(-1,1)) 


# In[58]:


print("Accuracy after CV :", accuracy_score(y_test, y_test_rnd_pred))
print("Pricision after CV:", precision_score(y_test, y_test_rnd_pred))
print("Recall after CV   :", recall_score(y_test, y_test_rnd_pred))
print("f1_score after CV :", f1_score(y_test, y_test_rnd_pred))


# # Using Decision Tree

# # train

# In[34]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[35]:


dt_clf=tree.DecisionTreeClassifier()
dt_clf.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))


# In[36]:


dt_clf.score(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))
y_dt_pred=dt_clf.predict(np.array(x_test).reshape(-1,1))


# In[37]:


y_dt_pred= dt_clf.predict(np.array(x_train).reshape(-1,1))


# In[38]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy after CV :", accuracy_score(y_train,y_dt_pred))
print("Pricision after CV:", precision_score(y_train, y_dt_pred))
print("Recall after CV   :", recall_score(y_train, y_dt_pred))
print("f1_score after CV :", f1_score(y_train, y_dt_pred))


# # test

# In[39]:


y_test_dt_pred= dt_clf.predict(np.array(x_test).reshape(-1,1)) 


# In[40]:


print("Accuracy after CV :", accuracy_score(y_test, y_test_dt_pred))
print("Pricision after CV:", precision_score(y_test, y_test_dt_pred))
print("Recall after CV   :", recall_score(y_test, y_test_dt_pred))
print("f1_score after CV :", f1_score(y_test, y_test_dt_pred))


# # Using KNN

# # train

# In[41]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))


# In[42]:


y_knn_pred= knn_clf.predict(np.array(x_train).reshape(-1,1))


# In[43]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy after CV :", accuracy_score(y_train, y_knn_pred))
print("Pricision after CV:", precision_score(y_train, y_knn_pred))
print("Recall after CV   :", recall_score(y_train, y_knn_pred))
print("f1_score after CV :", f1_score(y_train, y_knn_pred))


# # test

# In[44]:


y_test_knn_pred= knn_clf.predict(np.array(x_test).reshape(-1,1)) 


# In[45]:


print("Accuracy after CV :", accuracy_score(y_test, y_test_knn_pred))
print("Pricision after CV:", precision_score(y_test, y_test_knn_pred))
print("Recall after CV   :", recall_score(y_test, y_test_knn_pred))
print("f1_score after CV :", f1_score(y_test, y_test_knn_pred))


# In[ ]:




