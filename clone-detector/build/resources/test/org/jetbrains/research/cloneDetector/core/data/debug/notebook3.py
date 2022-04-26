#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
import numpy as np

def extract(data):
    f = open(data, 'rb')
    out = pickle.load(f)
    features = np.array(out['Patches'])
    labels = np.array(out['Corners'])
    f.close()
    return features,labels


# In[3]:


features,labels=extract('training.pkl')
print(labels[0])
t = labels[0].reshape((1,8))
print(t.shape)
print(t[0].shape)
print(features[0])


# In[6]:


features,corners = extract('unsup.pkl')


# In[7]:


print(features[0].shape)
print(corners[0].shape)


# In[ ]:




