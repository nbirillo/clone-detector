#!/usr/bin/env python
# coding: utf-8

# ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/deployment/accelerated-models/accelerated-models-training.png)

# Copyright (c) Microsoft Corporation. All rights reserved.
# 
# Licensed under the MIT License.

# # Training with the Azure Machine Learning Accelerated Models Service

# This notebook will introduce how to apply common machine learning techniques, like transfer learning, custom weights, and unquantized vs. quantized models, when working with our Azure Machine Learning Accelerated Models Service (Azure ML Accel Models).
# 
# We will use Tensorflow for the preprocessing steps, ResNet50 for the featurizer, and the Keras API (built on Tensorflow backend) to build the classifier layers instead of the default ImageNet classifier used in Quickstart. Then we will train the model, evaluate it, and deploy it to run on an FPGA.
# 
# #### Transfer Learning and Custom weights
# We will walk you through two ways to build and train a ResNet50 model on the Kaggle Cats and Dogs dataset: transfer learning only and then transfer learning with custom weights.
# 
# In using transfer learning, our goal is to re-purpose the ResNet50 model already trained on the [ImageNet image dataset](http://www.image-net.org/) as a basis for our training of the Kaggle Cats and Dogs dataset. The ResNet50 featurizer will be imported as frozen, so only the Keras classifier will be trained.
# 
# With the addition of custom weights, we will build the model so that the ResNet50 featurizer weights as not frozen. This will let us retrain starting with custom weights trained with ImageNet on ResNet50 and then use the Kaggle Cats and Dogs dataset to retrain and fine-tune the quantized version of the model.
# 
# #### Unquantized vs. Quantized models
# The unquantized version of our models (ie. Resnet50, Resnet152, Densenet121, Vgg16, SsdVgg) uses native float precision (32-bit floats), which will be faster at training. We will use this for our first run through, then fine-tune the weights with the quantized version. The quantized version of our models (i.e. QuantizedResnet50, QuantizedResnet152, QuantizedDensenet121, QuantizedVgg16, QuantizedSsdVgg) will have the same node names as the unquantized version, but use quantized operations and will match the performance of the model when running on an FPGA.
# 
# #### Contents
# 1. [Setup Environment](#setup)
# * [Prepare Data](#prepare-data)
# * [Construct Model](#construct-model)
#     * Preprocessor
#     * Classifier
#     * Model construction
# * [Train Model](#train-model)
# * [Test Model](#test-model)
# * [Execution](#execution)
#     * [Transfer Learning](#transfer-learning)
#     * [Transfer Learning with Custom Weights](#custom-weights)
# * [Create Image](#create-image)
# * [Deploy Image](#deploy-image)
# * [Test the service](#test-service)
# * [Clean-up](#cleanup)
# * [Appendix](#appendix)

# <a id="setup"></a>
# ## 1. Setup Environment
# #### 1.a. Please set up your environment as described in the [Quickstart](./accelerated-models-quickstart.ipynb), meaning:
# * Make sure your Workspace config.json exists and has the correct info
# * Install Tensorflow
# 
# #### 1.b. Download dataset into ~/catsanddogs 
# The dataset we will be using for training can be downloaded [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765). Download the zip and extract to a directory named 'catsanddogs' under your user directory ("~/catsanddogs"). 
# 
# 

# #### 1.c. Import packages

# In[ ]:


import os
import sys
import tensorflow as tf
import numpy as np
from keras import backend as K
import sklearn
import tqdm


# #### 1.d. Create directories for later use
# After you train your model in float32, you'll write the weights to a place on disk. We also need a location to store the models that get downloaded.

# In[ ]:


custom_weights_dir = os.path.expanduser("~/custom-weights")
saved_model_dir = os.path.expanduser("~/models")


# <a id="prepare-data"></a>
# ## 2. Prepare Data
# Load the files we are going to use for training and testing. By default this notebook uses only a very small subset of the Cats and Dogs dataset. That makes it run relatively quickly.

# In[ ]:


import glob
import imghdr
datadir = os.path.expanduser("~/catsanddogs")

cat_files = glob.glob(os.path.join(datadir, 'PetImages', 'Cat', '*.jpg'))
dog_files = glob.glob(os.path.join(datadir, 'PetImages', 'Dog', '*.jpg'))

# Limit the data set to make the notebook execute quickly.
cat_files = cat_files[:64]
dog_files = dog_files[:64]

# The data set has a few images that are not jpeg. Remove them.
cat_files = [f for f in cat_files if imghdr.what(f) == 'jpeg']
dog_files = [f for f in dog_files if imghdr.what(f) == 'jpeg']

if(not len(cat_files) or not len(dog_files)):
    print("Please download the Kaggle Cats and Dogs dataset form https://www.microsoft.com/en-us/download/details.aspx?id=54765 and extract the zip to " + datadir)    
    raise ValueError("Data not found")
else:
    print(cat_files[0])
    print(dog_files[0])


# In[ ]:


# Construct a numpy array as labels
image_paths = cat_files + dog_files
total_files = len(cat_files) + len(dog_files)
labels = np.zeros(total_files)
labels[len(cat_files):] = 1


# In[ ]:


# Split images data as training data and test data
from sklearn.model_selection import train_test_split
onehot_labels = np.array([[0,1] if i else [1,0] for i in labels])
img_train, img_test, label_train, label_test = train_test_split(image_paths, onehot_labels, random_state=42, shuffle=True)

print(len(img_train), len(img_test), label_train.shape, label_test.shape)


# <a id="construct-model"></a>
# ## 3. Construct Model
# We will define the functions to handle creating the preprocessor and the classifier first, and then run them together to actually construct the model with the Resnet50 featurizer in a single Tensorflow session in a separate cell.
# 
# We use ResNet50 for the featurizer and build our own classifier using Keras layers. We train the featurizer and the classifier as one model. We will provide parameters to determine whether we are using the quantized version and whether we are using custom weights in training or not.

# ### 3.a. Define image preprocessing step
# Same as in the Quickstart, before passing image dataset to the ResNet50 featurizer, we need to preprocess the input file to get it into the form expected by ResNet50. ResNet50 expects float tensors representing the images in BGR, channel last order. We've provided a default implementation of the preprocessing that you can use.
# 
# **Note:** Expect to see TF deprecation warnings until we port our SDK over to use Tensorflow 2.0.

# In[ ]:


import azureml.accel.models.utils as utils

def preprocess_images(scaling_factor=1.0):
    # Convert images to 3D tensors [width,height,channel] - channels are in BGR order.
    in_images = tf.placeholder(tf.string)
    image_tensors = utils.preprocess_array(in_images, 'RGB', scaling_factor)
    return in_images, image_tensors


# ### 3.b. Define classifier
# We use Keras layer APIs to construct the classifier. Because we're using the tensorflow backend, we can train this classifier in one session with our Resnet50 model.

# In[ ]:


def construct_classifier(in_tensor, seed=None):
    from keras.layers import Dropout, Dense, Flatten
    from keras.initializers import glorot_uniform
    K.set_session(tf.get_default_session())

    FC_SIZE = 1024
    NUM_CLASSES = 2

    x = Dropout(0.2, input_shape=(1, 1, int(in_tensor.shape[3]),), seed=seed)(in_tensor)
    x = Dense(FC_SIZE, activation='relu', input_dim=(1, 1, int(in_tensor.shape[3]),),
              kernel_initializer=glorot_uniform(seed=seed), bias_initializer='zeros')(x)
    x = Flatten()(x)
    preds = Dense(NUM_CLASSES, activation='softmax', input_dim=FC_SIZE, name='classifier_output',
                  kernel_initializer=glorot_uniform(seed=seed), bias_initializer='zeros')(x)
    return preds


# ### 3.c. Define model construction
# Now that the preprocessor and classifier for the model are defined, we can define how we want to construct the model. 
# 
# Constructing the model has these steps: 
# 1. Get preprocessing steps
# * Get featurizer using the Azure ML Accel Models SDK:
#     * import the graph definition
#     * restore the weights of the model into a Tensorflow session
# * Get classifier
# 

# In[ ]:


def construct_model(quantized, starting_weights_directory = None):
    from azureml.accel.models import Resnet50, QuantizedResnet50
    
    # Convert images to 3D tensors [width,height,channel]
    in_images, image_tensors = preprocess_images(1.0)

    # Construct featurizer using quantized or unquantized ResNet50 model
    if not quantized:
        featurizer = Resnet50(saved_model_dir)
    else:
        featurizer = QuantizedResnet50(saved_model_dir, custom_weights_directory = starting_weights_directory)

    features = featurizer.import_graph_def(input_tensor=image_tensors)
    
    # Construct classifier
    preds = construct_classifier(features)
    
    # Initialize weights
    sess = tf.get_default_session()
    tf.global_variables_initializer().run()

    featurizer.restore_weights(sess)

    return in_images, image_tensors, features, preds, featurizer


# <a id="train-model"></a>
# ## 4. Train Model

# In[ ]:


def read_files(files):
    """ Read files to array"""
    contents = []
    for path in files:
        with open(path, 'rb') as f:
            contents.append(f.read())
    return contents


# In[ ]:


def train_model(preds, in_images, img_train, label_train, is_retrain = False, train_epoch = 10, learning_rate=None):
    """ training model """
    from keras.objectives import binary_crossentropy
    from tqdm import tqdm
    
    learning_rate = learning_rate if learning_rate else 0.001 if is_retrain else 0.01
        
    # Specify the loss function
    in_labels = tf.placeholder(tf.float32, shape=(None, 2))   
    cross_entropy = tf.reduce_mean(binary_crossentropy(in_labels, preds))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    def chunks(a, b, n):
        """Yield successive n-sized chunks from a and b."""
        if (len(a) != len(b)):
            print("a and b are not equal in chunks(a,b,n)")
            raise ValueError("Parameter error")

        for i in range(0, len(a), n):
            yield a[i:i + n], b[i:i + n]

    chunk_size = 16
    chunk_num = len(label_train) / chunk_size

    sess = tf.get_default_session()
    for epoch in range(train_epoch):
        avg_loss = 0
        for img_chunk, label_chunk in tqdm(chunks(img_train, label_train, chunk_size)):
            contents = read_files(img_chunk)
            _, loss = sess.run([optimizer, cross_entropy],
                                feed_dict={in_images: contents,
                                           in_labels: label_chunk,
                                           K.learning_phase(): 1})
            avg_loss += loss / chunk_num
        print("Epoch:", (epoch + 1), "loss = ", "{:.3f}".format(avg_loss))
            
        # Reach desired performance
        if (avg_loss < 0.001):
            break


# <a id="test-model"></a>

# <a id="test-model"></a>
# ## 5. Test Model

# In[ ]:


def test_model(preds, in_images, img_test, label_test):
    """Test the model"""
    from keras.metrics import categorical_accuracy

    in_labels = tf.placeholder(tf.float32, shape=(None, 2))
    accuracy = tf.reduce_mean(categorical_accuracy(in_labels, preds))
    contents = read_files(img_test)

    accuracy = accuracy.eval(feed_dict={in_images: contents,
                                        in_labels: label_test,
                                        K.learning_phase(): 0})
    return accuracy


# <a id="execution"></a>
# ## 6. Execute steps
# You can run through the Transfer Learning section, then skip to Create AccelContainerImage. By default, because the custom weights section takes much longer for training twice, it is not saved as executable cells. You can copy the code or change cell type to 'Code'.
# 
# <a id="transfer-learning"></a>
# ### 6.a. Training using Transfer Learning

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Launch the training\ntf.reset_default_graph()\nsess = tf.Session(graph=tf.get_default_graph())\n\nwith sess.as_default():\n    in_images, image_tensors, features, preds, featurizer = construct_model(quantized=True)\n    train_model(preds, in_images, img_train, label_train, is_retrain=False, train_epoch=10, learning_rate=0.01)    \n    accuracy = test_model(preds, in_images, img_test, label_test)  \n    print("Accuracy:", accuracy)')


# #### Save Model

# In[ ]:


model_name = 'resnet50-catsanddogs-tl'
model_save_path = os.path.join(saved_model_dir, model_name)

tf.saved_model.simple_save(sess, model_save_path,
                               inputs={'images': in_images},
                               outputs={'output_alias': preds})

input_tensors = in_images.name
output_tensors = preds.name

print(input_tensors)
print(output_tensors)


# <a id="custom-weights"></a>
# ### 6.b. Traning using Custom Weights
# 
# Because the quantized graph defintion and the float32 graph defintion share the same node names in the graph definitions, we can initally train the weights in float32, and then reload them with the quantized operations (which take longer) to fine-tune the model.
# 
# First we train the model with custom weights but without quantization. Training is done with native float precision (32-bit floats). We load the training data set and batch the training with 10 epochs. When the performance reaches desired level or starts decredation, we stop the training iteration and save the weights as tensorflow checkpoint files. 

# #### Launch the training
# ```
# tf.reset_default_graph()
# sess = tf.Session(graph=tf.get_default_graph())
# 
# with sess.as_default():
#     in_images, image_tensors, features, preds, featurizer = construct_model(quantized=False)
#     train_model(preds, in_images, img_train, label_train, is_retrain=False, train_epoch=10)    
#     accuracy = test_model(preds, in_images, img_test, label_test)  
#     print("Accuracy:", accuracy)
#     featurizer.save_weights(custom_weights_dir + "/rn50", tf.get_default_session())
# ```

# #### Test Model
# After training, we evaluate the trained model's accuracy on test dataset with quantization. So that we know the model's performance if it is deployed on the FPGA.

# ```
# tf.reset_default_graph()
# sess = tf.Session(graph=tf.get_default_graph())
# 
# with sess.as_default():
#     print("Testing trained model with quantization")
#     in_images, image_tensors, features, preds, quantized_featurizer = construct_model(quantized=True, starting_weights_directory=custom_weights_dir)
#     accuracy = test_model(preds, in_images, img_test, label_test)      
#     print("Accuracy:", accuracy)
# ```

# #### Fine-Tune Model
# Sometimes, the model's accuracy can drop significantly after quantization. In those cases, we need to retrain the model enabled with quantization to get better model accuracy.

# ```
# if (accuracy < 0.93):
#     with sess.as_default():
#         print("Fine-tuning model with quantization")
#         train_model(preds, in_images, img_train, label_train, is_retrain=True, train_epoch=10)
#         accuracy = test_model(preds, in_images, img_test, label_test)        
#         print("Accuracy:", accuracy)
# ```

# #### Save Model

# ```
# model_name = 'resnet50-catsanddogs-cw'
# model_save_path = os.path.join(saved_model_dir, model_name)
# 
# tf.saved_model.simple_save(sess, model_save_path,
#                                inputs={'images': in_images},
#                                outputs={'output_alias': preds})
# 
# input_tensors = in_images.name
# output_tensors = preds.name
# ```

# <a id="create-image"></a>
# ## 7. Create AccelContainerImage
# 
# Below we will execute all the same steps as in the [Quickstart](./accelerated-models-quickstart.ipynb#create-image) to package the model we have saved locally into an accelerated Docker image saved in our workspace. To complete all the steps, it may take a few minutes. For more details on each step, check out the [Quickstart section on model registration](./accelerated-models-quickstart.ipynb#register-model).

# In[ ]:


from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.accel import AccelOnnxConverter
from azureml.accel import AccelContainerImage

# Retrieve workspace
ws = Workspace.from_config()
print("Successfully retrieved workspace:", ws.name, ws.resource_group, ws.location, ws.subscription_id, '\n')

# Register model
registered_model = Model.register(workspace = ws,
                                  model_path = model_save_path,
                                  model_name = model_name)
print("Successfully registered: ", registered_model.name, registered_model.description, registered_model.version, '\n', sep = '\t')

# Convert model
convert_request = AccelOnnxConverter.convert_tf_model(ws, registered_model, input_tensors, output_tensors)
if convert_request.wait_for_completion(show_output = False):
    # If the above call succeeded, get the converted model
    converted_model = convert_request.result
    print("\nSuccessfully converted: ", converted_model.name, converted_model.url, converted_model.version, 
          converted_model.id, converted_model.created_time, '\n')
else:
    print("Model conversion failed. Showing output.")
    convert_request.wait_for_completion(show_output = True)

# Package into AccelContainerImage
image_config = AccelContainerImage.image_configuration()
# Image name must be lowercase
image_name = "{}-image".format(model_name)
image = Image.create(name = image_name,
                     models = [converted_model],
                     image_config = image_config, 
                     workspace = ws)
image.wait_for_creation()
print("Created AccelContainerImage: {} {} {}\n".format(image.name, image.creation_state, image.image_location))


# <a id="deploy-image"></a>
# ## 8. Deploy image
# Once you have an Azure ML Accelerated Image in your Workspace, you can deploy it to two destinations, to a Databox Edge machine or to an AKS cluster. 
# 
# ### 8.a. Deploy to Databox Edge Machine using IoT Hub
# See the sample [here](https://github.com/Azure-Samples/aml-real-time-ai/) for using the Azure IoT CLI extension for deploying your Docker image to your Databox Edge Machine.
# 
# ### 8.b. Deploy to AKS Cluster

# #### Create AKS ComputeTarget

# In[ ]:


from azureml.core.compute import AksCompute, ComputeTarget

# Uses the specific FPGA enabled VM (sku: Standard_PB6s)
# Standard_PB6s are available in: eastus, westus2, westeurope, southeastasia
prov_config = AksCompute.provisioning_configuration(vm_size = "Standard_PB6s",
                                                    agent_count = 1,
                                                    location = "eastus")

aks_name = 'aks-pb6-tl'
# Create the cluster
aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_name, 
                                  provisioning_configuration = prov_config)


# Provisioning an AKS cluster might take awhile (15 or so minutes), and we want to wait until it's successfully provisioned before we can deploy a service to it. If you interrupt this cell, provisioning of the cluster will continue. You can re-run it or check the status in your Workspace under Compute.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'aks_target.wait_for_completion(show_output = True)\nprint(aks_target.provisioning_state)\nprint(aks_target.provisioning_errors)')


# #### Deploy AccelContainerImage to AKS ComputeTarget

# In[ ]:


get_ipython().run_cell_magic('time', '', "from azureml.core.webservice import Webservice, AksWebservice\n\n# Set the web service configuration (for creating a test service, we don't want autoscale enabled)\n# Authentication is enabled by default, but for testing we specify False\naks_config = AksWebservice.deploy_configuration(autoscale_enabled=False,\n                                                num_replicas=1,\n                                                auth_enabled = False)\n\naks_service_name ='my-aks-service-2'\n\naks_service = Webservice.deploy_from_image(workspace = ws,\n                                           name = aks_service_name,\n                                           image = image,\n                                           deployment_config = aks_config,\n                                           deployment_target = aks_target)\naks_service.wait_for_deployment(show_output = True)")


# <a id="test-service"></a>
# ## 9. Test the service
# 
# <a id="create-client"></a>
# ### 9.a. Create Client
# The image supports gRPC and the TensorFlow Serving "predict" API. We will create a PredictionClient from the Webservice object that can call into the docker image to get predictions. If you do not have the Webservice object, you can also create [PredictionClient](https://docs.microsoft.com/en-us/python/api/azureml-accel-models/azureml.accel.predictionclient?view=azure-ml-py) directly.
# 
# **Note:** If you chose to use auth_enabled=True when creating your AksWebservice.deploy_configuration(), see documentation [here](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice(class)?view=azure-ml-py#get-keys--) on how to retrieve your keys and use either key as an argument to PredictionClient(...,access_token=key).
# **WARNING:** If you are running on Azure Notebooks free compute, you will not be able to make outgoing calls to your service. Try locating your client on a different machine to consume it.

# In[ ]:


# Using the grpc client in AzureML Accelerated Models SDK
from azureml.accel import client_from_service

# Initialize AzureML Accelerated Models client
client = client_from_service(aks_service)


# <a id="serve-model"></a>
# ### 9.b. Serve the model
# Let's see how our service does on a few images. It may get a few wrong.

# In[ ]:


# Specify an image to classify
print('CATS')
for image_file in cat_files[:8]:
    results = client.score_file(path=image_file, 
                                 input_name=input_tensors, 
                                 outputs=output_tensors)
    result = 'CORRECT ' if results[0] > results[1] else 'WRONG '
    print(result + str(results))
print('DOGS')
for image_file in dog_files[:8]:
    results = client.score_file(path=image_file, 
                                 input_name=input_tensors, 
                                 outputs=output_tensors)
    result = 'CORRECT ' if results[1] > results[0] else 'WRONG '
    print(result + str(results))


# <a id="cleanup"></a>
# ## 10. Cleanup
# It's important to clean up your resources, so that you won't incur unnecessary costs.

# In[ ]:


aks_service.delete()
aks_target.delete()
image.delete()
registered_model.delete()
converted_model.delete()


# <a id="appendix"></a>
# ## 11. Appendix

# License for plot_confusion_matrix:
# 
# New BSD License
# 
# Copyright (c) 2007-2018 The scikit-learn developers.
# All rights reserved.
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission. 
# 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
# 
