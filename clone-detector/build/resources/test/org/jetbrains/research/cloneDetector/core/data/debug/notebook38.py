#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


# In[2]:


fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[3]:


print(np.max(x_train), np.min(x_train))


# In[4]:


# x = (x - u) / std

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train: [None, 28, 28] -> [None, 784]
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)


# In[5]:


print(np.max(x_train_scaled), np.min(x_train_scaled))


# In[6]:


hidden_units = [100, 100]
class_num = 10

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.int64, [None])

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                           hidden_unit,
                                           activation=tf.nn.relu)
logits = tf.layers.dense(input_for_next_layer,
                         class_num)
# last_hidden_output * W(logits) -> softmax -> prob
# 1. logit -> softmax -> prob
# 2. labels -> one_hot
# 3. calculate cross entropy
loss = tf.losses.sparse_softmax_cross_entropy(labels = y,
                                              logits = logits)
# get accuracy.
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


# In[7]:


print(x)
print(logits)


# In[8]:


# session

init = tf.global_variables_initializer()
batch_size = 20
epochs = 10
train_steps_per_epoch = x_train.shape[0] // batch_size
valid_steps = x_valid.shape[0] // batch_size

def eval_with_sess(sess, x, y, accuracy, images, labels, batch_size):
    eval_steps = images.shape[0] // batch_size
    eval_accuracies = []
    for step in range(eval_steps):
        batch_data = images[step * batch_size : (step+1) * batch_size]
        batch_label = labels[step * batch_size : (step+1) * batch_size]
        accuracy_val = sess.run(accuracy,
                                feed_dict = {
                                    x: batch_data,
                                    y: batch_label
                                })
        eval_accuracies.append(accuracy_val)
    return np.mean(eval_accuracies)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            batch_data = x_train_scaled[
                step * batch_size : (step+1) * batch_size]
            batch_label = y_train[
                step * batch_size : (step+1) * batch_size]
            loss_val, accuracy_val, _ = sess.run(
                [loss, accuracy, train_op],
                feed_dict = {
                    x: batch_data,
                    y: batch_label
                })
            print('\r[Train] epoch: %d, step: %d, loss: %3.5f, accuracy: %2.2f' % (
                epoch, step, loss_val, accuracy_val), end="")
        valid_accuracy = eval_with_sess(sess, x, y, accuracy,
                                        x_valid_scaled, y_valid,
                                        batch_size)
        print("\t[Valid] acc: %2.2f" % (valid_accuracy))


# In[ ]:




