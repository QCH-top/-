import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold

sonar = pd.read_csv(r'sonar.all-data', header=None, sep=',')
sonar1 = sonar.iloc[0:208, 0:61]#取208行，前61列（数据）
data = np.mat(sonar1)
#print(data)

for i in range(208):
    if sonar1.iloc[i, 60] == "R":
        sonar1.iloc[i, 60] = 0
    else:
        sonar1.iloc[i, 60] = 1#换成0，1来判断类别

data = np.mat(sonar1.iloc[:, 0:61])
data1 = np.mat(sonar1.iloc[:, 0:60])
#print(data)
#print(data1)
k = 2  # k为聚类的类别数
n = 208  # n为样本总个数
d = 60  # d为数据集的特征数


# k-means算法
def k_means():
    # 随机选k个初始聚类中心,聚类中心为每一类的均值向量
    m = np.zeros((k, d))#2行 60列零矩阵
    for i in range(k):
        m[i] = data1[np.random.randint(0, n)]
    
    # k_means聚类
    m_new = m.copy()
    #print("m_new = ",m_new)
    #print(m_new == m)
    t = 0
    while (1):
        # 更新聚类中心
        m[0] = m_new[0]
        m[1] = m_new[1]
        w1 = np.zeros((1, d+1))
        w2 = np.zeros((1, d+1))
        # 将每一个样本按照欧式距离聚类
        for i in range(n):#每一个样本跟两个聚类中心求欧氏距离
            distance = np.zeros(k)
            sample = data[i]#第i个样本
            for j in range(k):  # 将每一个样本与聚类中心比较
                distance[j] = np.linalg.norm(sample[:, 0:60] - m[j])#二范数
            category = distance.argmin()#种类 距离小的下标是0则是第一类
            if category == 0:
                w1 = np.row_stack((w1, sample))# 第一类的数据
            if category == 1:
                w2 = np.row_stack((w2, sample))

        # 新的聚类中心
        w1 = np.delete(w1, 0, axis=0)#数组合并后会多出一个0删除0
        w2 = np.delete(w2, 0, axis=0)
        m_new[0] = np.mean(w1[:, 0:60], axis=0)
        m_new[1] = np.mean(w2[:, 0:60], axis=0)

        # 聚类中心不再改变时，聚类停止
        if (m[0] == m_new[0]).all() and (m[1] == m_new[1]).all():
            break

        print(t)
        t += 1

        w = np.vstack((w1, w2))
        label1 = np.zeros((len(w1), 1))
        label2 = np.ones((len(w2), 1))
        label = np.vstack((label1, label2))
        label = np.ravel(label)#合并为一维矩阵
      #  print(label）
        test_PCA(w, label)#pac降维
        plot_PCA(w, label)
    return w1, w2


def test_PCA(*data):
    X, Y = data
    pca = decomposition.PCA(n_components=None)#自动选择降维维度
    pca.fit(X)#训练模型


# print("explained variance ratio:%s"%str(pca.explained_variance_ratio_))

def plot_PCA(*data):
    X, Y = data
    pca = decomposition.PCA(n_components=2)#降维2个维度
    pca.fit(X)
    X_r = pca.transform(X)#执行降维
    #   print(X_r)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0.33, 0.33, 0.33),)
    for label, color in zip(np.unique(Y), colors):
        position = Y == label
        ax.scatter(X_r[position, 0], X_r[position, 1], label="category=%d" % label, color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
   # plt.show()


if __name__ == '__main__':
    w1, w2 = k_means()
    accuracy = 0

    l1 = []
    l2 = []
    for i in range(w1.shape[0]):
        l1.append(w1[i, 4])
    l2.append(l1.count(0))
    l2.append(l1.count(1))

    l3 = np.mat(l2)
    count = l3.argmax()
    for i in range(w1.shape[0]):
        if w1[i, 60] == count:
            accuracy += 1

    count2 = 0
    if count == 0:
        count2=1
    else:
        count2=0

    for i in range(w2.shape[0]):
        if w2[i, 60] == count2:
            accuracy += 1


    # print(w1)

    accuracy /= 150
    print("准确度为：")
    print("%.2f" % accuracy)

    print("第一类的聚类样本数为：")
    print(w1.shape[0])
    # print(w2)
    print("第二类的聚类样本数为：")
    print(w2.shape[0])

    plt.show()
    
