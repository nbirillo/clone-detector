#!/usr/bin/env python
# coding: utf-8

# In[21]:


from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')


# In[22]:


train_data_list = []
train_label_list = []
test_data_list = []
test_label_list = []

scaler = StandardScaler()
# scaler = MinMaxScaler()

for i in range(1, 11):
    mat_data = loadmat("data/train/"+str(i)+".mat")
    train_data_list.append(scaler.fit_transform(mat_data['de_feature']))
    train_label_list.append(mat_data['label'])

for i in range(11, 14):
    mat_data = loadmat("data/test/"+str(i)+".mat")
    test_data_list.append(scaler.fit_transform(mat_data['de_feature']))
    test_label_list.append(mat_data['label'])

train_datas = np.concatenate(train_data_list)
train_labels = np.concatenate(train_label_list)
test_datas = np.concatenate(test_data_list)
test_labels = np.concatenate(test_label_list)

pca = PCA(n_components=2)
pca_train_datas = pca.fit_transform(train_datas)
pca_test_datas = pca.fit_transform(test_datas)
pca_train_data_list = [pca.fit_transform(x) for x in train_data_list]
pca_test_data_list = [pca.fit_transform(x) for x in test_data_list]


# In[23]:


def draw_pca(data: np.array, label: np.array, name=None, size=0.5):
    label = label.squeeze()
    assert len(data) == len(label)
    assert data.shape[1] == 2
    optioned = [False]*4
    for idx in range(len(data)):
        point = data[idx]
        if label[idx] == 0:
            if not optioned[0]:
                plt.scatter(point[0], point[1], c='#3B77A8', label='negtive', s=size)
                optioned[0] = True
            else:
                plt.scatter(point[0], point[1], c='#3B77A8', s=size)
        elif label[idx] == 1:
            if not optioned[1]:
                plt.scatter(point[0], point[1], c='#FFDB50', label='neutral', s=size)
                optioned[1] = True
            else:
                plt.scatter(point[0], point[1], c='#FFDB50', s=size)
        elif label[idx] == 2:
            if not optioned[2]:
                plt.scatter(point[0], point[1], c='#F37726', label='positive', s=size)
                optioned[2] = True
            else:
                plt.scatter(point[0], point[1], c='#F37726', s=size)
        elif label[idx] == 3:
            if not optioned[3]:
                plt.scatter(point[0], point[1], c='red', label='fear', s=size)
                optioned[3] = True
            else:
                plt.scatter(point[0], point[1], c='red', s=size)
        
    plt.legend()
#     if name is not None:
#         plt.savefig("D:\\TC文件夹\\1_2019-2020第2学期\\工科创4J\\hw02\\LaTex\\" + name + ".png", dpi=250)
    plt.show()


# In[24]:


def compute_acc(pred, labels=test_labels):
    return (pred == labels.squeeze()).sum()/len(pred)


# In[25]:


mysvm = svm.SVC(gamma='scale', C=10, decision_function_shape='ovo', max_iter=300000, probability=True)
mysvm.fit(train_datas, train_labels)


# In[26]:


for i in range(3):
    pred = mysvm.predict(test_data_list[i])
    acc = compute_acc(pred, test_label_list[i])
    print("Person %d, acc: %.5f" %(i+1, acc))


# In[27]:


pred = mysvm.predict(test_datas)
print("Average acc: %.5f" % (compute_acc(pred),))


# In[ ]:




