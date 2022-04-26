#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[12]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import ceil
from CIoTS import *
import time


# In[13]:


max_ps = list(range(4, 9, 2))
test_ps = list(range(2, 11, 2))
runs = 20
dimensions = 4
data_length = 10000
alpha = 0.05


# In[14]:


f1_scores = pd.DataFrame(columns=['true_p', 'p', 'mean_f1', 'std_f1'])
runtimes = pd.DataFrame(columns=['true_p', 'p', 'mean_time', 'std_time'])


# In[15]:


for p in max_ps:
    incoming_edges = 3 #max(ceil(dimensions*p/3), 1)
    
    f1 = {p_test: [] for p_test in test_ps}
    runtime = {p_test: [] for p_test in test_ps}
    
    for run in range(runs):
        generator = CausalTSGenerator(dimensions=dimensions, max_p=p, data_length=data_length, incoming_edges=incoming_edges)
        ts = generator.generate()
            
        for p_test in test_ps:
            start_time = time.time()
            predicted_graph = pc_chen(partial_corr_test, ts, p_test, alpha)
            exec_time = time.time()-start_time
            
            f1[p_test].append(evaluate_edges(generator.graph, predicted_graph)['f1-score'])
            runtime[p_test].append(exec_time)
            print('done: p='+ str(p) + ' run='+str(run+1) + ' p_test='+str(p_test) + ' exec_time='+str(exec_time))
    
    for p_test in test_ps:
        f1_scores = f1_scores.append({'true_p': p, 'p': p_test, 'mean_f1': np.mean(f1[p_test]),
                                      'std_f1': np.std(f1[p_test])}, ignore_index=True)
        runtimes = runtimes.append({'true_p': p, 'p': p_test, 'mean_time': np.mean(runtime[p_test]),
                                    'std_time': np.std(runtime[p_test])}, ignore_index=True)


# In[16]:


f1_scores


# In[17]:


plt.figure(figsize=(8,8))
plt.title('f1 scores for differen p')
plt.xlabel('assumed p')
plt.ylabel('mean f1')
handles = []
labels = []
for p in max_ps:
    plt.errorbar(x=f1_scores.loc[f1_scores['true_p']==p, 'p'],
                 y=f1_scores.loc[f1_scores['true_p']==p, 'mean_f1'],
                 yerr=f1_scores.loc[f1_scores['true_p']==p, 'std_f1'],
                 label='true p='+str(p))
plt.legend()
plt.show()


# In[18]:


plt.figure(figsize=(8,8))
plt.title('runtime for different p')
plt.xlabel('assumed p')
plt.ylabel('runtime in s')
handles = []
labels = []
for p in max_ps:
    plt.errorbar(x=runtimes.loc[runtimes['true_p']==p, 'p'],
                 y=runtimes.loc[runtimes['true_p']==p, 'mean_time'],
                 yerr=runtimes.loc[runtimes['true_p']==p, 'std_time'],
                 label='true p='+str(p))
plt.legend()
plt.show()


# In[ ]:




