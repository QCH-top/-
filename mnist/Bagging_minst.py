#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import gzip
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn import datasets, decomposition, manifold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[9]:


data = pd.read_csv(r'mnist_train.csv',header = None)
data1 = pd.read_csv(r'mnist_test.csv',header = None)
data = np.array(data)
data1 = np.array(data1)
train_label = data[:,0]  #训练集标签
train_images = data[:,1:784]  #训练集图片
test_label = data1[:,0]  #训练集标签
test_images = data1[:,1:784]  #训练集图片


# In[10]:


all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = []  # 总数据对应的类标签
all_data_set = train_images
all_data_label = train_label

n_components = 16 #降为16维度
pca = decomposition.PCA(n_components=n_components, svd_solver='auto',
          whiten=True).fit(all_data_set)
# PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
# X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
y = np.array(all_data_label)

n_estimators = [10,11]
criterion_test_names = ["gini", "entropy"]#测试 系数与熵
                                          #分类树: 基尼系数 最小的准则 在sklearn中可以选择划分的默认原则      
                                          #决策树：criterion:默认是’gini’系数，也可以选择信息增益的熵’entropy’


# In[11]:


#RF使用了CART决策树作为弱学习器，
#在使用决策树的基础上，
#RF对决策树的建立做了改进
#RF通过随机选择节点上的一部分样本特征，
#这个数字小于n
#假设为nsub，
#然后在这些随机选择的nsub个样本特征中，
#选择一个最优的特征来做决策树的左右子树划分。
#这样进一步增强了模型的泛化能力。
#总的来说，随机森林是在将bagging方法的基学习器确定为决策树，并且在决策树的训练过程中引入了随机属性选择


# In[12]:


def RandomForest(n_estimators,criterion):
    # 十折交叉验证计算出平均准确率
    # 交叉验证，随机取
    kf = KFold(n_splits=10, shuffle=True)
    precision_average = 0.0
    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}  # 自动穷举出最优的C参数
    # clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced', gamma=param),
    #                    param_grid)

    parameter_space = {
        "min_samples_leaf": [2, 4, 6], }#参数空间   
    clf = GridSearchCV(RandomForestClassifier(random_state=14,n_estimators = n_estimators,criterion = criterion), parameter_space, cv=5)
    for train, test in kf.split(X):
        clf = clf.fit(X[train], y[train])
        # print(clf.best_estimator_)
        test_pred = clf.predict(X[test])
        # print classification_report(y[test], test_pred)
        # 计算平均准确率
        precision = 0
        for i in range(0, len(y[test])):
            if (y[test][i] == test_pred[i]):
                precision = precision + 1
        precision_average = precision_average + float(precision) / len(y[test])
    precision_average = precision_average / 10
    print (u"准确率为" + str(precision_average))
    return precision_average


# In[13]:


print(n_estimators)


# In[ ]:


for criterion in criterion_test_names:
    print("criterion",criterion)
    x_label = []
    y_label = []
    for i in range(10,15):
        print(i)
        y_label.append(RandomForest(i,criterion))
        x_label.append(i)
    plt.plot(x_label, y_label, label=criterion)
    time += 1
# print("done in %0.3fs" % (time() - t0))
plt.xlabel("criterion")
plt.ylabel("Precision")
plt.title('Different  Contrust')
plt.legend()
plt.show()


# In[ ]:




