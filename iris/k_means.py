import matplotlib.pyplot as plt
from random import sample
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import datasets
from sklearn.datasets import load_iris
#导入鸢尾花数据集
#以二维数据为例 假设k=2，X为鸢尾花数据集前两维
iris = load_iris()
X = iris.data[:,0:2] ##表示我们只取特征空间中的前两个维度 X类型是np.array
print(len(X))
#绘制数据分布图 显示前两维数据
plt.scatter(X[:, 0], X[:, 1], c = "red", marker='o', label='data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
 
#从X中随机选择k个样本作为初始“簇中心”向量： μ(1),μ(2),...,,μ(k)
#随机获得两个数据
n = 3 #表示n个聚类
u = sample(X.tolist(),n) #随机选择n个X中的向量作为聚类中心
max_iter = 0 #记录迭代次数
u_before = u
 
while max_iter<5:
    #簇分配过程
    c = []
    print(u_before,u)
    for i in range(len(X)):
        min = 1000
        index = 0
        for j in range(n):
            vec1 = X[i]
            vec2 = u[j]
            dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
            if dist<min:
                min = dist
                index =j
        c.append(index)
        # print(i,"------",c[i])
    #移动聚类中心
    for j in range(n):
        sum = np.zeros(2)  # 定义n为零向量 此处n为2
        count = 0  # 统计不同类别的个数
        for i in range(len(X)):
            if c[i]==j:
                sum = sum+X[i]
                count = count+1
        u[j] = sum/count
 
    print(max_iter,"------------",u)
    #设置迭代次数
    max_iter = max_iter + 1
print(np.array(c))
label_pred = np.array(c)
#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()