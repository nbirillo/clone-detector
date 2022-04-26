#!/usr/bin/env python
# coding: utf-8

# # Hypothesis 21, 23 & 24

# In[1]:


from pyspark.sql import SparkSession, functions as F, types as T
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


spark = SparkSession.builder.getOrCreate()


# In[3]:


orders_df = spark.read                       .option('quote', '\"')                       .option('escape', '\"')                       .csv('./dataset/olist_orders_dataset.csv', header=True, multiLine=True, inferSchema=True)

order_items_df = spark.read                       .option('quote', '\"')                       .option('escape', '\"')                       .csv('./dataset/olist_order_items_dataset.csv', header=True, multiLine=True, inferSchema=True)

orders_df.printSchema()

order_items_df.printSchema()


# In[4]:


order_items_df.columns


# In[5]:


new_df = orders_df.join(order_items_df.groupBy('order_id')
                        .agg(F.sum('freight_value')
                             .alias('total_freight')), 
                        how='left', 
                        on='order_id')


# In[6]:


new_df.limit(5).toPandas()


# In[7]:


new_df = new_df.join(order_items_df.groupBy('order_id')
                     .agg(F.sum('price')
                          .alias('total_price')), 
                     how='left', 
                     on='order_id')

new_df.limit(5).toPandas()


# In[9]:


new_df = new_df.join(order_items_df.groupBy('order_id')
                     .agg(F.max('order_item_id')
                          .alias('total_items')),
                     how='left', 
                     on='order_id')

new_df = new_df.filter(F.col('total_price').isNotNull())                .filter(F.col('total_freight').isNotNull())


# In[10]:


new_df.limit(10).toPandas()


# In[11]:


new_df.filter(F.col('order_id') == '1b15974a0141d54e36626dca3fdc731a').toPandas()


# ## Hypothesis 21: The order's itens quantity is directly proportional to the full order value (price + freight)

# In[12]:


aux_df = new_df.select(F.col('total_items').alias('itens_qnty').cast(T.IntegerType()),
                       F.col('total_price'),
                       F.col('total_freight'))
aux_df.limit(10).show()


# In[13]:


aux_df.filter(F.col('order_id') == '1b15974a0141d54e36626dca3fdc731a').show()


# In[14]:


myUdf = F.udf(lambda x,y: float(x)+float(y), T.DoubleType())

aux_df = aux_df.withColumn('order_price', F.round(myUdf('total_price', 'total_freight'), 2))


# In[15]:


aux_df.limit(10).show()


# In[16]:


aux_df.stat.corr('itens_qnty', 'order_price')


# In[17]:


plt.figure(figsize=(16, 6))
sns.scatterplot(x='itens_qnty', y='order_price', data=aux_df.toPandas())


# ## Conclusion H21
# 
# The hypothesis 21 is **invalid**, as the full price of the order has no direct correlation with the order itens quantity

# ## Hypothesis 23: The freight value is directly proportional to the order's time of delivery

# In[18]:


aux_df = new_df.filter(F.col('order_delivered_customer_date').isNotNull())

aux_df = new_df.select(F.col('total_freight'),
                       F.col('order_purchase_timestamp').alias('purchase'),
                       F.col('order_delivered_customer_date').alias('deliver'))

aux_df = aux_df.withColumn('order_duration', F.datediff(F.col('deliver'), F.col('purchase')))
aux_df = aux_df.filter(F.col('order_duration').isNotNull())
aux_df = aux_df.drop('purchase', 'deliver')

aux_df.show()


# In[19]:


aux_df.stat.corr('total_freight', 'order_duration')


# ## Conclusion H23
# 
# The hypothesis 23 is **invalid**, as there is nearly no correlation between the freight value and the time of delivery

# ## Hypothesis 24: The freight value is directly proportional to the order's items quantity

# In[20]:


aux_df = new_df.select(F.col('total_items'),
                       F.col('total_freight'))

aux_df.show()


# In[21]:


aux_df.stat.corr('total_items', 'total_freight')


# ## Conclusion H24
# 
# The hypothesis 24 is **valid**, as there is a considerable relation between the freight value and the order's items quantity

# In[ ]:




