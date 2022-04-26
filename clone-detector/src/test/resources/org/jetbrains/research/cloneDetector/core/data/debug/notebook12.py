#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/saurabh-1991/DeepLearning-Algorithms/blob/master/detecting_type_of_clothes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import tensorflow as tf
import keras
from PIL import Image

from matplotlib.pyplot import imshow
import numpy as np


# In[2]:


#Importing Fashion_MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print("Training Images = ",len(train_images))
print("Training Labels = ",len(train_labels))

imshow(train_images[0])


# In[3]:


Training_Images = train_images / 255.0
Testing_Images =  test_images / 255.0
print (type(Training_Images))
print (np.shape(Training_Images))


# 

# In[ ]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# In[10]:


model.compile(optimizer =tf.train.AdamOptimizer(),
              loss ='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(Training_Images,train_labels,epochs = 5)


# In[8]:


test_score = model.evaluate(test_images,test_labels)
print('Test loss {:.4f}, accuracy {:.2f}%'.format(test_score[0], test_score[1] * 100))

