#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
m=[1,1,1]
# n=m.tostring()
ss=[str(i) for i in m]
n=",".join(ss)
print(n)


# In[9]:


lidar = np.array([[[1, 2, 3], [3, 4, 5],[1,1,1]],[[1, 2, 3], [3, 4, 5],[1,1,1]],[[1, 2, 3], [3, 4, 5],[1,1,1]]])


# In[10]:


x = lidar[:, :, 0].reshape(-1)
y = lidar[:, :, 1].reshape(-1)
z = lidar[:, :, 2].reshape(-1)
cloud = np.stack((x, y, z))


# In[11]:


cloud


# In[18]:


import re
sss="POLYGON Z ((22.68176642 114.36814367 0.00000000, 22.68177278 114.36816370 0.00000000, 22.68176088 114.36817013 0.00000000, 22.68175855 114.36815157 0.00000000, 22.68176642 114.36814367 0.00000000))"
tmp=re.findall(r"\d+\.?\d*", sss)
print(tmp)
temp_point =np.array([tmp[i:i + 3] for i in range(0, len(tmp), 3)], dtype=float)
print(temp_point)
x = temp_point[ :, 0].reshape(-1)
y = temp_point[:, 1].reshape(-1)
z = temp_point[:, 2].reshape(-1)
cloud = np.stack((x, y, z))
cloud

