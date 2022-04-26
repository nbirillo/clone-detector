#!/usr/bin/env python
# coding: utf-8

# # RobustRegression
# * `simi15` -- Huber Regression
# * `simi16` -- Theil-Sen Regression
# 

# In[1]:


# add path
import sys; import os; sys.path.append(os.path.realpath("../"))

# general hyperparameter optimization settings
from seasalt import (select_the_best, refit_model) 
from seasalt.si import (cv_settings, scorerfun, print_scores)
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


# demo datasets
from datasets.demo2 import X_train, Y_train, fold_ids, X_valid, Y_valid, meta as meta_data
#meta_data


# In[3]:


# model implementations
#from potpourri.simi15 import model, hyper, meta  # Huber Regression
from potpourri.simi16 import model, hyper, meta  # Theil-Sen Regression
meta


# ## Train

# In[4]:


get_ipython().run_cell_magic('time', '', 'rscv = RandomizedSearchCV(**{\'estimator\': model, \'param_distributions\': hyper}, **cv_settings)\nrscv.fit(X = X_train, y = Y_train)  # Run CV\n\nbestparam, summary = select_the_best(rscv)  # find the "best" parameters\nbestmodel = refit_model(model, bestparam, X_train, Y_train)  # Refit the "best" model')


# In[5]:


#rscv.cv_results_


# ## Evaluate

# In[7]:


print("Infer/predict on validation set")
Y_pred = bestmodel.predict(X_valid)

print("\nOut of sample score")
print(scorerfun(Y_valid, Y_pred))

print("\nOut of sample score (Other metrics)")
print_scores(Y_pred, Y_valid)

print("\nBest model parameters")
print(bestparam)

print("\nIn-sample scores and model variants (from CV)")
summary


# ### Parameters

# In[8]:


bestmodel.steps[1][1].coef_


# ### Target vs Predicted

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(6,6))
plt.scatter(y=Y_pred, x=Y_valid);
#plt.scatter(x=np.log(Y_pred), y=np.log(Y_valid));
plt.xlabel('target');
plt.ylabel('predicted');


# ## Debug, Memory, Misc

# In[10]:


#del summary
#locals()
get_ipython().run_line_magic('whos', '')


# In[ ]:




