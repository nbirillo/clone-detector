#!/usr/bin/env python
# coding: utf-8

# ## Load Data

# In[37]:


import os
import numpy as np
import csv
import pandas as pd
# Display progress logs on stdout
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def readLabelData(path):
    fo = open(path, "r")
    data = fo.readlines();
    fo.close()
    res = []
    for x in data:
        x = x.rstrip('\n')
        id, tweet = x.split('\t')
        res.append([int(id), tweet])
    return res

def readTweets(path):
    fo = open(path, "r")
    data = fo.readlines();
    fo.close()
    res = []
    for x in data:
        x = x.rstrip('\n')
        res.append(x)
    return res


# In[38]:


f = open("data/tweets/test_tweets.txt","rb")
test_tweets = []
while True:
    line = f.readline()
    if not line:
        break
    else:
        try:
            #print(line.decode('utf8'))
            test_tweets.append(line.decode('utf8'))
            #为了暴露出错误，最好此处不print
        except:
#             continue
            test_tweets.append("47,USER_d12c6a27,\"@USER_a23336a5 hey. How's the day off going?\",?")
#             print(str(line))
f.close()
print(len(test_tweets))
for i in range(len(test_tweets)):
    temp1 = test_tweets[i].strip().split('"')
    temp2 = temp1[0].split(',')
    temp3 = temp1[-1].strip(',')
    temp4 = temp1[1:-1]
    temp5 = " ".join(x for x in temp4)
    test_tweets[i] = [temp2[0], temp2[1], temp5, temp3]
print(len(test_tweets))


# In[39]:


import pandas as pd
df_test = pd.DataFrame(test_tweets)
df_test.columns = ['tweet-id', 'user-id', 'tweet', 'class']
df_test[['tweet']].to_csv('out_test.txt', index=False, header=False, sep='\t')
df_test.head()


# In[40]:


df_test[['tweet-id']].to_csv('out_test_id.txt', index=False, header=False, sep='\t')


# ### Load Train

# In[41]:


f = open("data/tweets/train_tweets.txt","rb")
train_tweets = []
while True:
    line = f.readline()
    if not line:
        break
    else:
        try:
            #print(line.decode('utf8'))
            train_tweets.append(line.decode('utf8'))
            #为了暴露出错误，最好此处不print
        except:
            continue
#             print(str(line))
f.close()


# In[42]:


len(train_tweets)


# ### Load Dev

# In[43]:


f = open("data/tweets/dev_tweets.txt","rb")
dev_tweets = []
while True:
    line = f.readline()
    if not line:
        break
    else:
        try:
            #print(line.decode('utf8'))
            dev_tweets.append(line.decode('utf8'))
            #为了暴露出错误，最好此处不print
        except:
            continue
#             print(str(line))
f.close()


# In[44]:


len(dev_tweets)


# ### Load Test

# In[45]:


f = open("data/tweets/test_tweets.txt","rb")
test_tweets = []
while True:
    line = f.readline()
    if not line:
        break
    else:
        try:
            #print(line.decode('utf8'))
            test_tweets.append(line.decode('utf8'))
            #为了暴露出错误，最好此处不print
        except:
            continue
#             print(str(line))
f.close()


# In[46]:


len(test_tweets)


# In[47]:


myTrain = train_tweets + dev_tweets
for i in range(len(myTrain)):
    temp1 = myTrain[i].strip().split('"')
    temp2 = temp1[0].split(',')
    temp3 = temp1[-1].strip(',')
    temp4 = temp1[1:-1]
    temp5 = " ".join(x for x in temp4)
    myTrain[i] = [temp2[0], temp2[1], temp5, temp3]
len(myTrain)


# In[48]:


df = pd.DataFrame(myTrain)
df.columns = ['tweet-id', 'user-id', 'tweet', 'class']
df[['class', 'tweet']].to_csv('out.txt', index=False, header=False, sep='\t')
df.head()


# In[49]:


from nltk.corpus import stopwords
from textblob import Word

def Preprocess(train):
    train['tweet'] = train['tweet'].str.replace('@[\w]*', '')
    train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
    stop = stopwords.words('english')
    train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    


# In[50]:


Preprocess(df)
df.head()


# In[51]:


freq = pd.Series(' '.join(df['tweet']).split()).value_counts()


# In[52]:


from textblob import TextBlob
#NewYork, California, Georgia, total
wordCount = [0, 0, 0, 0]
for index, row in df.iterrows():
    tokens = TextBlob(row['tweet']).words
    class_index = -1
    if (row['class'] == "NewYork"): 
        class_index = 0
    elif (row['class'] == "California"):
        class_index = 1
    elif (row['class'] == "Georgia"): 
        class_index = 2
    else:
        print(row['class'])
    for token in tokens:
        wordCount[class_index] += 1
        wordCount[3] += 1


# In[53]:


wordCount


# In[54]:


dictWithFreq = {}

for index, row in df.iterrows():
    tokens = TextBlob(row['tweet']).words
    class_index = -1
    if (row['class'] == "NewYork"): 
        class_index = 0
    elif (row['class'] == "California"):
        class_index = 1
    elif (row['class'] == "Georgia"): 
        class_index = 2
    else:
        print(row['class'])
    for token in tokens:
        if (token in dictWithFreq): 
            dictWithFreq[token][class_index] += 1 
            dictWithFreq[token][3] +=1
        else :
            dictWithFreq[token] = [0, 0, 0, 0]
            dictWithFreq[token][class_index] +=1
            dictWithFreq[token][3] +=1
            


# In[55]:


freq.head()


# In[56]:


dictWithFreq['rt']


# In[57]:


df.head()


# In[58]:


pNewYork = []
pCalifornia = []
pGeorgia = []

for index, row in df.iterrows():
    tokens = TextBlob(row['tweet']).words
    pNewYork.append(0)
    pCalifornia.append(0)
    pGeorgia.append(0)
    for token in tokens:
        if (token in dictWithFreq):
            if (dictWithFreq[token][3] < 50): continue
            # p(city|tweet) = p(city|word) * p(word)
            pNewYork[-1] += (dictWithFreq[token][0] / wordCount[3])
            pCalifornia[-1] += (dictWithFreq[token][1] / wordCount[3])
            pGeorgia[-1] += (dictWithFreq[token][2] / wordCount[3]) 


# In[59]:


df['pNewYork'] = pNewYork
df['pCalifornia'] = pCalifornia
df['pGeorgia'] = pGeorgia
df.head()


# In[68]:


dSum = {}
for index, row in df.iterrows():
    user_id = row['user-id']
    arr = row.tolist()[4:]
    if user_id in dSum:
        for i in range(len(arr)):
            dSum[user_id][i] += arr[i]
    else:
        dSum[user_id] = arr


# In[69]:


dfOut = df[['tweet-id', 'user-id','class']]
dfOut.head()


# In[70]:


dfOut['outArr'] = dfOut['user-id'].apply(lambda x: dSum[x])
dfOut.head()


# In[72]:


pCity = []
for index, row in dfOut.iterrows():
    pNewYork = row['outArr'][0]
    pCalifornia = row['outArr'][1]
    pGeorgia = row['outArr'][2]
    if (pNewYork >= pCalifornia and pNewYork >= pGeorgia):
        pCity.append('NewYork')
    elif (pCalifornia >= pNewYork and pCalifornia >= pGeorgia):
        pCity.append('California')
    elif (pGeorgia >= pNewYork and pGeorgia >= pCalifornia):
        pCity.append('Georgia')
    else:
        print('ERR')


# In[73]:


df['pCity'] = pCity
df.head()


# In[74]:


true = 0
for index, row in df.iterrows():
    if row['class'] == row['pCity']:
        true += 1
        
true / df.shape[0]


# ## the First Part: 
# accracy: 0.20763679266965968

# In[32]:


class2Int = []

for index, row in df.iterrows():
    tokens = TextBlob(row['tweet']).words
    class_index = -1
    if (row['class'] == "NewYork"): 
        class_index = 0
    elif (row['class'] == "California"):
        class_index = 1
    elif (row['class'] == "Georgia"): 
        class_index = 2
    else:
        print(row['class'])
    class2Int.append(class_index)


# In[33]:


df['class2Int'] = class2Int
df.head()


# ## save data

# In[35]:


df[['pNewYork','pCalifornia','pGeorgia']].to_csv('outPre.txt', index=False, header=False, sep='\t')


# In[ ]:




