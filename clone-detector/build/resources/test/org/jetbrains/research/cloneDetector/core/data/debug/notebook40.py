#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'png', 'retina'}")


# In[2]:


taxi = pd.read_csv("test/test.csv")


# In[3]:


taxi["count"] = 1


# In[4]:


import datetime as dt


# In[5]:


taxi.tail()


# ## pickup_datetime 살피기

# In[6]:


# pd.to_datetime을 해줘야 pickup_datetime coulmn의 data type이 datetime으로 됨.
pickup_datetime_dt = pd.to_datetime(taxi["pickup_datetime"])


# In[7]:


# trip_duration, count를 포함하고, pickup_datetime의 data로만 이루어진 dataframe(taxi_df1)을 만든다.
taxi_df1 = taxi.loc[:, ["trip_duration", "count"]]
taxi_df1["pickup_datetime"] = pickup_datetime_dt


# In[8]:


# pickup datetime 중 year, month, day
taxi_df1.loc[:, "pickup_date"] = taxi_df1["pickup_datetime"].dt.date
# pickup datetime 중 month만 가져와서 새로 column을 만듦.
taxi_df1.loc[:, "pickup_month"] = taxi_df1["pickup_datetime"].dt.month
# pickup datetime 중 hour만 가져와서 새로 column을 만듦.
taxi_df1.loc[:, "pickup_hour"] = taxi_df1["pickup_datetime"].dt.hour
# pickup datetime 중 요일만 가져와서 새로 column을 만듦.
# 월요일은 0이고, 일요일은 6임.
taxi_df1.loc[:, "pickup_weekday"] = taxi_df1["pickup_datetime"].dt.weekday


# In[9]:


taxi_df1.tail()


# - year 살피기

# In[10]:


print(taxi_df1["pickup_datetime"].dt.year.min())
print(taxi_df1["pickup_datetime"].dt.year.max())


# - month 살피기

# In[11]:


print(taxi_df1["pickup_month"].min())
print(taxi_df1["pickup_month"].max())


# In[12]:


taxi_month_1 = taxi_df1.loc[:, ["pickup_month", "count"]]
taxi_month_1.groupby("pickup_month").sum()


# - hour 살피기

# In[15]:


print(taxi_df1["pickup_hour"].min())
print(taxi_df1["pickup_hour"].max())


# In[16]:


taxi_hour_1 = taxi_df1.loc[:, ["pickup_hour", "count"]]
taxi_hour_1.groupby("pickup_hour").sum()


# - 요일 살피기

# In[19]:


print(taxi_df1["pickup_weekday"].min())
print(taxi_df1["pickup_weekday"].max())


# In[20]:


taxi_weekday_1 = taxi_df1.loc[:, ["pickup_weekday", "count"]]
taxi_weekday_1.groupby("pickup_weekday").sum()


# ### EDA of pickup_datetime in test data
# - year : 2016년 
# - month : 1~6월
# - hour : 0~23시
# - weekday : 월요일~일요일
