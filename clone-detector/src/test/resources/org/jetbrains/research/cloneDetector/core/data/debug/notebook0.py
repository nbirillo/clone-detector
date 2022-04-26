#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/feud72/hands_on_ml/blob/master/9%EC%9E%A5_%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C_%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# 공통 
import numpy as np 
import os

def reset_graph(seed=42): 
  tf.reset_default_graph() 
  tf.set_random_seed(seed) 
  np.random.seed(seed)

# 맷플롯립 설정 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력 
plt.rcParams['font.family'] = 'NanumBarunGothic' 
plt.rcParams['axes.unicode_minus'] = False

get_ipython().run_line_magic('tensorflow_version', '1.x')


# In[ ]:


import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2


# In[47]:


sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)

print(result)


# In[ ]:


sess.close()


# In[ ]:


with tf.Session() as sess:
  x.initializer.run()
  y.initializer.run()
  result = f.eval()


# In[50]:


result


# In[ ]:


init = tf.global_variables_initializer()


# In[52]:


sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)


# In[ ]:


sess.close()


# In[54]:


reset_graph()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()


# In[55]:


graph = tf.Graph()
with graph.as_default():
  x2 = tf.Variable(2)

x2.graph is graph


# In[56]:


x2.graph is tf.get_default_graph()


# In[57]:


w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
  print(y.eval())
  print(z.eval())


# In[58]:


with tf.Session() as sess:
  y_val, z_val = sess.run([y, z])
  print(y_val)
  print(z_val)


# In[ ]:


import numpy as np
from sklearn.datasets import fetch_california_housing

reset_graph()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]


# In[ ]:


X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
  theta_value = theta.eval()


# In[61]:


theta_value


# In[62]:


X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)


# In[63]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]


# In[65]:


print(scaled_housing_data_plus_bias.mean(axis=0))
print(scaled_housing_data_plus_bias.mean(axis=1))
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)


# In[66]:


reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch % 100 == 0:
      print("에포크", epoch, "MSE =", mse.eval())
    sess.run(training_op)

  best_theta = theta.eval()


# In[ ]:


def my_func(a, b):
  z = 0
  for i in range(100):
    z = a * np.cos(z + i) + z * np.sin(b - i)
  return z


# In[68]:


my_func(0.2, 0.3)


# In[ ]:


reset_graph()

a = tf.Variable(0.2, name="a")
b = tf.Variable(0.3, name="b")
z = tf.constant(0.0, name="z0")
for i in range(100):
  z = a * tf.cos(z + i) + z * tf.sin(b - i)

grads = tf.gradients(z, [a, b])
init = tf.global_variables_initializer()


# In[70]:


with tf.Session() as sess:
  init.run()
  print(z.eval())
  print(sess.run(grads))


# In[ ]:


reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")


# In[ ]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


# In[73]:


init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch % 100 == 0:
      print("에포크", epoch, "MSE = ", mse.eval())
    sess.run(training_op)
  
  best_theta = theta.eval()

print("best_theta:")
print(best_theta)


# In[75]:


reset_graph()

A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5

with tf.Session() as sess:
  B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
  B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8,  9]]})

print(B_val_1)

print(B_val_2)


# In[ ]:


reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")


# In[ ]:


batch_size = 100
n_batches = int(np.ceil(m / batch_size))


# In[ ]:


theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42))
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()


# In[ ]:


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


# In[ ]:


def fetch_batch(epoch, batch_index, batch_size):
  np.random.seed(epoch * n_batches + batch_index)
  indices = np.random.randint(m, size=batch_size)
  X_batch = scaled_housing_data_plus_bias[indices]
  y_batch = housing.target.reshape(-1, 1)[indices]
  return X_batch, y_batch

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    for batch_index in range(n_batches):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
      sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

  best_theta = theta.eval()


# In[81]:


best_theta


# In[ ]:


reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")


# In[85]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch % 100 == 0:
      print("에포크", epoch, "MSE = ", mse.eval())
      save_path = saver.save(sess, "/tmp/mymodel.ckpt")
    sess.run(training_op)

  best_theta = theta.eval()
  save_path = saver.save(sess, "/tmp/my_model_final.ckpt")


# In[86]:


best_theta


# In[87]:


with tf.Session() as sess:
  saver.restore(sess, "/tmp/my_model_final.ckpt")
  best_theta_restored = theta.eval()


# In[88]:


np.allclose(best_theta, best_theta_restored)


# In[ ]:


saver = tf.train.Saver({"weights": theta})


# In[ ]:


reset_graph()

saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")
theta = tf.get_default_graph().get_tensor_by_name("theta:0")


# In[92]:


with tf.Session() as sess:
  saver.restore(sess, "/tmp/my_model_final.ckpt")
  best_theta_restored = theta.eval()


# In[93]:


np.allclose(best_theta, best_theta_restored)


# In[97]:


from tensorboardcolab import *


# In[99]:


tbc=TensorBoardColab(startup_waiting_time=30)


# In[ ]:


train_writer = tbc.get_writer();
train_writer.add_graph(sess.graph)


# In[ ]:


train_writer.flush()
tbc.close()

