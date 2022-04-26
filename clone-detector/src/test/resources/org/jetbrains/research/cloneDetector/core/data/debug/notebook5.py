#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'cypher')
import json
import random


# In[24]:


geotweets = get_ipython().run_line_magic('cypher', 'match (n:tweet) where n.coordinates is not null return n.tid, n.lang, n.country, n.name, n.coordinates, n.created_at')


# In[25]:


geotweets = geotweets.get_dataframe()
geotweets.head()


# In[26]:


json.loads(geotweets.ix[1]["n.coordinates"])[0][0]


# In[27]:


def get_random_coords(df):
    lats = []
    lons = []
    for row in df.iterrows():
        row = row[1]
        coords = json.loads(row["n.coordinates"])[0]
        lat1 = coords[0][0]
        lat2 = coords[2][0]
        lon1 = coords[0][1]
        lon2 = coords[1][1]
        ran_lat = random.uniform(lat1, lat2)
        ran_lon = random.uniform(lon1, lon2)
        lats.append(ran_lat)
        lons.append(ran_lon)
    df["lat"] = lats
    df["lon"] = lons
    return df


# In[28]:


df = get_random_coords(geotweets)


# In[29]:


geotweets.columns = ["Id", "Lang", "Country", "City", "Coords", "Time", "Lon", "Lat"]


# In[32]:


geotweets["Label"] = "tweet"


# In[33]:


geotweets.head()


# In[34]:


geotweets.to_csv("data/geotweets.csv")


# In[2]:


edges_query = """match (t:tweet)-[:USES]->(h:hashtag) where t.coordinates is not null with h.tagid as hashtag, t.tid as tweet return hashtag, tweet
"""


# In[7]:


geotweet_edges = get_ipython().run_line_magic('cypher', 'match (t:tweet)-[:USES]->(h:hashtag) where t.coordinates is not null with h.tagid as hashtag, t.tid as tweet return tweet, hashtag')


# In[8]:


geotweet_edges = geotweet_edges.get_dataframe()


# In[9]:


geotweet_edges.head()


# In[11]:


geotweet_edges.columns = ["Source", "Target"]


# In[22]:


geotweet_edges.to_csv("data/geoedges.csv")


# In[39]:


geoedges_nohash = get_ipython().run_line_magic('cypher', 'match (t:tweet)--(n:tweet) where t.coordinates is not null and n.coordinates is not null return t.tid as Source, n.tid as Target')


# In[40]:


geoedges_nohash = geoedges_nohash.get_dataframe()


# In[41]:


len(geoedges_nohash)


# In[43]:


geoedges_nohash.to_csv("data/geoedges_nohash.csv")


# In[ ]:





# In[56]:


geohash = get_ipython().run_line_magic('cypher', 'match (t:tweet)-[r:USES]->(h:hashtag) where t.coordinates is not null with distinct h.tagid as Id, h.hashtag as Label, count(r) as deg return Id, Label order by deg desc limit 10')


# In[57]:


geohash = geohash.get_dataframe()
geohash.head()


# In[58]:


labels = geohash["Label"].map(lambda x: "#" + x)


# In[59]:


geohash["Label"] = labels


# In[60]:


geohash.head()


# In[61]:


geohash.to_csv("data/geotags.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


edges = get_ipython().run_line_magic('cypher', 'match (t:tweet)-[:USES]-(h:hashtag {hashtag: "paris"}) where t.coordinates is not null return h.hashtag, collect(t.tid) ')


# In[4]:


import itertools
import networkx as nx


# In[5]:


edges = edges.get_dataframe()


# In[6]:


edges["collect(t.tid)"] = edges["collect(t.tid)"].map(lambda x: list(itertools.combinations(x, 2)))


# In[7]:


edges.head()


# In[8]:


el = list(itertools.chain.from_iterable(edges["collect(t.tid)"]))


# In[9]:


len(el)


# In[8]:


el[1]


# In[9]:


len(el)


# In[9]:


g = nx.Graph(el)


# In[34]:


len(geotweet_edges)


# In[ ]:




