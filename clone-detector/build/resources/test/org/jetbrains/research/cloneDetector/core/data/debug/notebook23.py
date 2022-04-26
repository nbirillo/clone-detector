#!/usr/bin/env python
# coding: utf-8

# In[34]:


#Predict the target class. 
#import dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#classifies data set to analyze
#read in the csv file
df = pd.read_csv('Classified Data',index_col=0)


# In[4]:


#check the head of the dataframe
df.head()


# In[7]:


#want to predict the 'TARGET CLASS'
#standardize data to the same scale
from sklearn.preprocessing import StandardScaler


# In[8]:


#create an instance of StandardScaler like for any other ML algorithm
scaler = StandardScaler()


# In[10]:


#fit scaler to the data minus the Target Class
scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[11]:


#transform the data
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[12]:


#display the scaled feature
scaled_features
#displays the array of features


# In[14]:


#feature dataframe
#include all columns except the last one which is the 'TARGET CLASS'
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[15]:


#check the head of the df_feat
df_feat.head()


# In[18]:


#Data is ready to be put into a ML algo
#train test split
from sklearn.model_selection import train_test_split


# In[19]:


X= scaled_features
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


#USE KNN model to determine whether someone will be inside the 'Target Class'


# In[20]:


#With k=1
from sklearn.neighbors import KNeighborsClassifier


# In[21]:


knn = KNeighborsClassifier(n_neighbors = 1)


# In[22]:


#fit the data 
knn.fit(X_train,y_train)


# In[23]:


#grab predictions to evaluate
#pass in the test data 
pred = knn.predict(X_test)


# In[24]:


#displays where people fall based on the features
pred


# In[26]:


#import classification matirix and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix


# In[27]:


#print
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[31]:


#find a better K value using the elbow methd
#plot to see the best k value
error_rate=[]

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[36]:


#plot the figure
plt.figure(figsize =(10,6))
plt.plot(range(1,40),error_rate, color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
#from the graph you can see the lower the K value the higher the error rate


# In[37]:


#print with a higher K value
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[41]:


#print with a higher K value
knn = KNeighborsClassifier(n_neighbors=37)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




