#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[283]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import itertools as it
import os.path as osp

import helpers_06

get_ipython().run_line_magic('matplotlib', 'inline')


# # AlexNet

# ## Grab and Load the Pre-existing Weights

# Next week, we are going to do full *transfer learning*.  But, for this week, we are going to take a different, simpler strategy.  In our model building up to this point, we've initialized the weights in our networks randomly (from a truncated normal distribution or something similar).  We're going to do much, much better today.  Instead of starting with random weights and training them, we're going to make use of weights that someone already did the hard work of training!  
# 
# So, we're going to have a two part strategy:
#   1.  Create the architecture of AlexNet (that matches the weights someone else learned)
#   2.  Use those learned weights to initialize our network.
#   
# Then, boom, we have a working AlexNet without having to train it.  If you are unimpressed, realized that the inital training of the AlexNet in the acdemic paper took about six days on a machine with two highend GPUs.  We are really saving computational work here!
# 
# Getting the weights is an exercise in frustration, so we've insulated you from that process with a helper.  Here goes:

# In[284]:


pretrained_weights, readable_labels = helpers_06.fetch_alexnet_weights_and_classes()


# ## Exercise

# You can use your Python-fu to investigate `pretrained_weights` and `readable_labels`.  In particular:
#   1.  What type of thing is `pretrained_weights`?  
#   2.  What is its structure?
#   3.  If I tell you that the inner-most values come as `W,b` pairs, can you get the *shapes* of the coefficients?
#   4.  What's in `readable_labels`?

# ### Solution

# Weights are stored as a dictionary of lists, with each entry storing the trained weight and bias values.

# In[285]:


print(pretrained_weights['conv1'][0].shape)
print(pretrained_weights['conv1'][1].shape)
print(pretrained_weights['conv2'][0].shape)
print(pretrained_weights['conv2'][1].shape)
print(pretrained_weights['conv3'][0].shape)
print(pretrained_weights['conv3'][1].shape)
print(pretrained_weights['conv4'][0].shape)
print(pretrained_weights['conv4'][1].shape)
print(pretrained_weights['conv5'][0].shape)
print(pretrained_weights['conv5'][1].shape)
print(pretrained_weights['fc6'][0].shape)
print(pretrained_weights['fc6'][1].shape)
print(pretrained_weights['fc7'][0].shape)
print(pretrained_weights['fc7'][1].shape)
print(pretrained_weights['fc8'][0].shape)
print(pretrained_weights['fc8'][1].shape)


# In[286]:


print(readable_labels)


# These values tell us the size of the network we need to create.  Fortunately, since we can use NumPy arrays (directly) to create Tensors, we don't have to copy these shapes down anywhere! 

# ## Defining Layers

# ##### Convolution Layers

# Having the weights ahead of time is very helpful -- if we can make use of them!  We've got to do something a little bit different.  Here's an old way that we recreated a convolutional layer.  Remember that we set the weights and biases to random values and a constant, respectively.

# In[287]:


def random_init_conv_layer(incoming, num_kernels, kernel_sz, 
                           strides=[1, 1], padding='SAME',
                           bval=0.01, 
                           activation_fn=tf.nn.relu, 
                           name=None):
    prev_outshape = incoming.shape.dims[-1].value
    kshape = kernel_sz + [prev_outshape, num_kernels]

    fan_in = np.prod(incoming.shape[1:]).value
    xavier_stddev = np.sqrt(2.0 / fan_in)
    
    strides = [1] + strides + [1]
    with tf.variable_scope(name, 'conv_layer'):
        w = tf.Variable(tf.truncated_normal(kshape, stddev=xavier_stddev), name='kernel')
        b = tf.Variable(tf.constant(bval, shape=[num_kernels]), name='bias')
        conv = tf.nn.conv2d(incoming, w, strides, padding, name='conv')
        z = tf.nn.bias_add(conv, b)
        return z if activation_fn is None else activation_fn(z)


# ## Exercise

# Create a convolution layer that uses explictly given NumPy arrays to initialize its weights and biases.

# In[288]:


# pt stands for "pre-trained"
def pt_conv_layer(incoming, init_weight, init_bias,
                  strides=[1, 1], padding='SAME',
                  activation_fn=tf.nn.relu, 
                  name=None):
    'expects to be given numpy arrays or tensors for weight/bias'
    strides = [1] + strides + [1]
    with tf.variable_scope(name, 'conv_layer'):
        W = tf.get_variable("kernel", initializer=init_weight)
        b = tf.get_variable("bias", initializer=init_bias)
        conv = tf.nn.conv2d(incoming, W, strides, padding, name='conv')
        z = tf.nn.bias_add(conv, b)
        a = activation_fn(z)
        return a


# ##### Split Convolution
# To mimic the architecture of AlexNet, we can split the convolution layer.  When AlexNet was trained for real, this is the portion that got divided between two GPUs.  This isn't 100% necessary for us to use, because we are just filling in weights.  But it shows how you would do the splitting, if you ever have a need for it.

# In[289]:


# pt stands for "pre-trained"
def pt_split_conv_layer(incoming, init_weight, init_bias, 
                        strides=[1, 1], padding='SAME',
                        activation_fn=tf.nn.relu, 
                        name=None):
    strides = [1] + strides + [1]
    with tf.variable_scope(name, 'split_conv_layer'):
        w = tf.get_variable('kernel', initializer=init_weight)
        b = tf.get_variable('bias', initializer=init_bias)
        
        if True: # ignoring the split part for now
            #i_a, i_b = tf.split(incoming, 2, 3, name='split_inputs')
            #w_a, w_b = tf.split(w, 2, 3, name='split_weights')
            i_a, i_b = tf.split(incoming, num_or_size_splits=2, axis=3, name='split_inputs')
            w_a, w_b = tf.split(w, num_or_size_splits=2, axis=3, name='split_weights')
            
            print(i_a, w_a)
            conv_a = tf.nn.conv2d(i_a, w_a, strides, padding, name='conv_a')
            conv_b = tf.nn.conv2d(i_b, w_b, strides, padding, name='conv_b')
            merge = tf.concat([conv_a, conv_b], 3)
        else:
            conv = tf.nn.conv2d(incoming, w, strides, padding, name='conv')        
        
        z = tf.nn.bias_add(merge, b)
        a = activation_fn(z) if activation_fn is not None else z
        return a


# ##### AlexNet Utility Layers

# Two more layers that are useful for AlexNet.  Here, we make use of TensorFlow's built-in `relu_layer` to make a fully connected layer.  If you want to know where the magic constants in `alex_lrn_layer` come from, see:
#   * http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward_newtf.py

# In[290]:


# pt stands for "pre-trained"
def pt_alex_fc_layer(incoming, init_weights, init_biases, name=None):
    'expects to be given numpy arrays or tensors for weight/bias'
    with tf.name_scope('fully_connected'):
        return tf.nn.relu_layer(incoming, init_weights, init_biases, name=name)
    
def alex_lrn_layer(inputs):
    'lrn with some magic constants'
    return tf.nn.local_response_normalization(inputs, 2, 1.0, 2e-05, 0.75)


# ##### Pooling Layers

# This is the same as last week's, with a minor convenience tweak to prepend/append `1`s to the stride for us.

# In[291]:


def pool_layer(incoming, ksize, strides, padding='VALID',
                pool_fn=tf.nn.max_pool, name=None):
    # pooling layer with stride padding
    ksize = [1] + ksize + [1]
    strides = [1] + strides + [1]    
    with tf.variable_scope(name, 'pool_layer'):
        return pool_fn(incoming, ksize, strides, padding)
    


# In[292]:


def flatten(incoming, name=None):
    flat_shape = [-1, np.prod(incoming.shape[1:]).value]
    return tf.reshape(incoming, flat_shape)


# ## Defining the Main AlexNet Graph

# Here's the architecture of the main AlexNet component:
# 
# ![](images/alexnet.png)
# 

# And, here are the layer weight sizes (say that three times fast!) that we loaded above:
# 
#     conv1: (11, 11, 3, 96)
#     conv2: (5, 5, 48, 256)
#     conv3: (3, 3, 256, 384)
#     conv4: (3, 3, 192, 384)
#     conv5: (3, 3, 192, 256)
#     fc6: (9216, 4096)
#     fc7: (4096, 4096)
#     fc8: (4096, 1000)

# ## Exercise

# Try to put the architecture and the sizes together to make your own AlexNet.  If you can do it wihtout the following hints, you are a rockstart.
#   1.  The layer structure is (note, lrn and pool don't have weights):
#     *  conv, lrn, pool
#     *  split, lrn, pool
#     *  conv, split, split, pool, flat
#     *  full, full, and a special fully connected: `tf.nn.xw_plus_b`.
#   2.  The size and strides for the pool layers are `[3,3]` and `[2,2]` respectively.
#   3.  You can find the stride for the first conv in the diagram.
#   4.  If you are very stubborn and don't want to look at the solution, see here for more hints:
#     *   * http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward_newtf.py

# In[293]:


def alexnet(images, init_weights):
    # FILL ME IN
    with tf.name_scope('layer1'):
        conv_1 = pt_conv_layer(images, init_weights['conv1'][0], init_weights['conv1'][1],
                               strides=[4, 4],
                               padding='VALID')
        lrn_1 = alex_lrn_layer(conv_1)
        max_pool_1 = pool_layer(lrn_1, ksize=[3, 3], strides=[2, 2])
        print(max_pool_1.shape)
    
    with tf.name_scope('layer2'):
        conv_2 = pt_split_conv_layer(max_pool_1, init_weights['conv2'][0], init_weights['conv2'][1])
        lrn_2 = alex_lrn_layer(conv_2)
        max_pool_2 = pool_layer(lrn_2, ksize=[3, 3], strides=[2, 2])
        print(max_pool_2.shape)
    
    with tf.name_scope('layer3'):
        #conv_3 = pt_split_conv_layer(max_pool_2, init_weights['conv3'][0], init_weights['conv3'][1])
        conv_3 = pt_conv_layer(max_pool_2, init_weights['conv3'][0], init_weights['conv3'][1])
        print(conv_3.shape)
    
    with tf.name_scope('layer4'):
        conv_4 = pt_split_conv_layer(conv_3, init_weights['conv4'][0], init_weights['conv4'][1])
        print(conv_4.shape)
        
    with tf.name_scope('layer5'):
        conv_5 = pt_split_conv_layer(conv_4, init_weights['conv5'][0], init_weights['conv5'][1])
        max_pool_5 = pool_layer(conv_5, ksize=[3, 3], strides=[2, 2])
        print(max_pool_5.shape)
        
    with tf.name_scope('layer6'):
        flat_shape = [-1, np.prod(max_pool_5.shape[1:]).value]
        flat_6 = tf.reshape(max_pool_5, flat_shape)
        fc_6 = tf.nn.relu_layer(flat_6, init_weights['fc6'][0], init_weights['fc6'][1])
        
    with tf.name_scope('layer7'):
        fc_7 = tf.nn.relu_layer(fc_6, pretrained_weights['fc7'][0], pretrained_weights['fc7'][1])
        
    with tf.name_scope('layer8'):
        fc_8 = tf.nn.relu_layer(fc_7, pretrained_weights['fc8'][0], pretrained_weights['fc8'][1])
        
    return fc_8


# ## The "Loaded Weights" AlexNet Model

# We've now got all the pieces we need to make use of an AlexNet model made from previously learned (and shared with us) weights.  We'll sneak one other piece into our `class`:
# 
#     def save(s):
#         with s.graph.as_default():
#             tf.train.Saver().save(s.session, osp.join(_dir, "alexnet"))
# 
# This will let us do: `mm.save('saved_models')` to save our model.

# In[294]:


class AlexNet_Model:
    def __init__(self, img_height, img_width, init_weights):  # non-standard, for abbreviation
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                # consider two possibilities here:
                # 1.  use a fixed image size [img_height x img_width]
                # s.images = tf.placeholder(tf.float32, 
                #                           shape=[None, img_height, img_width, 3], name="images")
                # 2.  use variable image sizes -and- rescale the images to the same size
                #     in a rescaling step (see name_scope('rescale')) ... note, one "batch"
                #     of images must all be the same size of image
                self.images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="images")

                # note, we don't need labels, b/c the work is already done for us (no fitting/training)
                
            with tf.name_scope('rescale'):
                self.scaled_images = tf.image.resize_images(self.images, [img_height, img_width])
                

            #self.logits = alexnet(self.scaled_images, init_weights, debug=True)
            self.logits = alexnet(self.scaled_images, init_weights)
                        
            with tf.name_scope('prediction'):
                self.softmax    = tf.nn.softmax(self.logits, name="softmax")
                self.prediction = tf.cast(tf.arg_max(self.softmax, 1), tf.int32) # FIXME: unnecessary cast?

            self.init = tf.global_variables_initializer()
            
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

    def save(self, _dir):
        # fixme:  save(session) versus export_meta_graph
        helpers_06.mkdir(_dir)
        with self.graph.as_default():
            tf.train.Saver().save(self.session, osp.join(_dir, "alexnet"))
        
        
    def predict(self, test_dict):
        pred, prob = self.session.run([self.prediction, self.softmax], feed_dict=test_dict)
        return pred, prob


# ## Train our AlexNet

# In[295]:


# from above:
# pretrained_weights, readable_labels = helpers_06.fetch_alexnet_weights_and_classes()
mm = AlexNet_Model(227, 227, init_weights=pretrained_weights)


# ## Test our AlexNet

# ##### Grab Some Test Images

# In[296]:


img_names = ['dog', 'laska', 'poodle']
base_url = "http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/"
local_dir = "data/alexnet/"

for img_name in img_names:
    helpers_06.maybe_download("{}.png".format(img_name), base_url, local_dir)


# In[297]:


from scipy.misc import imread
import os.path as osp
img = imread(osp.join(local_dir, 'dog.png'))[:,:,:3]
plt.imshow(img)
plt.axis('off');


# ## Exercise

# Use our reconstituted model to label this nice dog.  You'll need two things to do it:
#   1.  We wanted indices into the labels (`readable_labels` from way above).  With the output `probs`, `top5_indices = np.argsort(probs, 1)[0][-5:]`.
#   2.  With only one image, we have to fake a batch.  Do this: `img_batch = np.expand_dims(img, 0)`.  Then, you can use `img_batch` as if it were multiple images.

# ### Solution

# In[300]:


result = mm.predict(img)


# # Save our Model

# In[298]:


mm.save('saved_models')


# # Load our Model

# And here is how we can bring that saved model back.  We'll play with a "reloaded" model more next week

# In[299]:


# create a placeholder graph to "rehydrate" our freeze-dried AlexNet
old_alex_graph = tf.Graph()
with old_alex_graph.as_default():
    # importing the graph will populate new_alex_graph
    saver = tf.train.import_meta_graph("saved_models/alexnet.meta")


# # Practice with Optimizers

# To get some practice with optimizers, go back to one of your MNIST (or CIFAR) models.  The Optimizers are listed here:
#   * https://www.tensorflow.org/api_guides/python/train
# 
# Experiment with using several different optimizers and see:
#   1.  learning performance
#   2.  convergance rate
#   3.  processing time (per step/epoch).
