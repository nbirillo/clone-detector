#!/usr/bin/env python
# coding: utf-8

# ## 实现混淆矩阵，精准率和召回率

# In[1]:


import numpy as np
from sklearn import datasets


# In[2]:


digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)


# In[4]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)


# In[5]:


y_log_predict = log_reg.predict(X_test)


# In[6]:


def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

TN(y_test, y_log_predict)


# In[7]:


def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

FP(y_test, y_log_predict)


# In[8]:


def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

FN(y_test, y_log_predict)


# In[9]:


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

TP(y_test, y_log_predict)


# In[10]:


def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

confusion_matrix(y_test, y_log_predict)


# In[11]:


def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
    
precision_score(y_test, y_log_predict)


# In[12]:


def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0
    
recall_score(y_test, y_log_predict)


# ### scikit-learn中的混淆矩阵，精准率和召回率

# In[13]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_log_predict)


# In[14]:


from sklearn.metrics import precision_score

precision_score(y_test, y_log_predict)


# In[15]:


from sklearn.metrics import recall_score

recall_score(y_test, y_log_predict)

