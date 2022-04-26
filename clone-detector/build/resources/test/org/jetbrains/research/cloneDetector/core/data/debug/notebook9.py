#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Informacion faltante
# 
# Aprendamos algunos metodos importantes para la manipulacion de datos nulos

# In[1]:


#librerias
import numpy as np
import pandas as pd


# In[2]:


df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})


# In[3]:


df


# In[4]:


df.dropna()


# In[5]:


df.dropna(axis=1)


# In[6]:


df.dropna(thresh=2)


# In[7]:


df.fillna(value='FILL VALUE')


# In[8]:


df['A'].fillna(value=df['A'].mean())

