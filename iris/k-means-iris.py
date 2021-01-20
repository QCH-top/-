import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold

iris = pd.read_csv(r'iris.data', header=None, sep=',')
iris1 = iris.iloc[0:150, 0:5]
for i in range(150):
    if iris1.iloc[i, 4] == "Iris-setosa":
        iris1.iloc[i, 4] = 0
    elif iris1.iloc[i, 4] == "Iris-versicolor":
        iris1.iloc[i, 4] = 1
    else:
        iris1.iloc[i, 4] = 2
data = np.mat(iris1.iloc[:, 0:5])
data1 = np.mat(iris1.iloc[:, 0:4])
k = 3  # k为聚类的类别数
n = 150  # n为样本总个数
d = 4  # t为数据集的特征数


# k-means算法
def k_means():
    # 随机选k个初始聚类中心,聚类中心为每一类的均值向量
    m = np.zeros((k, d))   # m = (3, 4)
    for i in range(k):
        m[i] = data1[np.random.randint(0, 10)]
    # k_means聚类
    m_new = m.copy()

    t = 0
    while (1):
        # 更新聚类中心
        m[0] = m_new[0]
        m[1] = m_new[1]
        m[2] = m_new[2]

        w1 = np.zeros((1, d+1))  # 引进标签
        w2 = np.zeros((1, d+1))
        w3 = np.zeros((1, d+1))

        for i in range(n):
            distance = np.zeros(3)
            sample = data[i]
            for j in range(k):  # 将每一个样本与聚类中心比较
                distance[j] = np.linalg.norm(sample[:, 0:4] - m[j])    # 求范数，默认计算欧式距离
            category = distance.argmin()
            if category == 0:
                w1 = np.row_stack((w1, sample))  # 行添加
            if category == 1:
                w2 = np.row_stack((w2, sample))
            if category == 2:
                w3 = np.row_stack((w3, sample))

        # 新的聚类中心
        w1 = np.delete(w1, 0, axis=0)
        w2 = np.delete(w2, 0, axis=0)
        w3 = np.delete(w3, 0, axis=0)
        m_new[0] = np.mean(w1[:, 0:4], axis=0)
        m_new[1] = np.mean(w2[:, 0:4], axis=0)
        m_new[2] = np.mean(w3[:, 0:4], axis=0)

        # 聚类中心不再改变时，聚类停止
        if (m[0] == m_new[0]).all() and (m[1] == m_new[1]).all() and (m[2] == m_new[2]).all():
            break
        # print(t)
        t += 1
        w = np.vstack((w1, w2))
        w = np.vstack((w, w3))
       '''
        # 画出每一次迭代的聚类效果图
  
        label1 = np.zeros((len(w1), 1))
        label2 = np.ones((len(w2), 1))
        label3 = np.zeros((len(w3), 1))
        for i in range(len(w3)):
            label3[i, 0] = 2
        label = np.vstack((label1, label2))
        label = np.vstack((label, label3))
        label = np.ravel(label)
        test_PCA(w, label)
        plot_PCA(w, label)
'''
    return w1, w2, w3, t


def test_PCA(*data):
    X, Y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(X)


# print("explained variance ratio:%s"%str(pca.explained_variance_ratio_))

def plot_PCA(*data):
    X, Y = data
    pca = decomposition.PCA(n_components=50)
    pca.fit(X)
    X_r = pca.transform(X)
    #   print(X_r)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0.3, 0.3, 0.3), (0, 0.3, 0.7),)
    for label, color in zip(np.unique(Y), colors):
        position = Y == label
        #      print(position)
        ax.scatter(X_r[position, 0], X_r[position, 1], label="category=%d" % label, color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    # plt.show()


if __name__ == '__main__':
    w1, w2, w3, t = k_means()
    accuracy = 0

    l1 = []
    l2 = []
    for i in range(w1.shape[0]):  # 标签投票，得票多数判断为该类
        l1.append(w1[i, 4])
    l2.append(l1.count(0))
    l2.append(l1.count(1))
    l2.append(l1.count(2))
    l3 = np.mat(l2)
    count = l3.argmax()
    for i in range(w1.shape[0]):
        if w1[i, 4] == count:
            accuracy += 1

    l1 = []
    l2 = []
    for i in range(w2.shape[0]):
        l1.append(w2[i, 4])
    l2.append(l1.count(0))
    l2.append(l1.count(1))
    l2.append(l1.count(2))
    l3 = np.mat(l2)
    count = l3.argmax()
    for i in range(w2.shape[0]):
        if w2[i, 4] == count:
            accuracy += 1

    l1 = []
    l2 = []
    for i in range(w3.shape[0]):
        l1.append(w3[i, 4])
    l2.append(l1.count(0))
    l2.append(l1.count(1))
    l2.append(l1.count(2))
    l3 = np.mat(l2)
    count = l3.argmax()
    for i in range(w3.shape[0]):
        if w3[i, 4] == count:
            accuracy += 1
    print(accuracy)
    accuracy /= 150  # 纯度计算
    print("准确度为：")
    print("%.2f"%accuracy)
    print("迭代次数为：")
    print(t)
    print("第一类的聚类样本数为：")
    print(w1.shape[0])
    print("第二类的聚类样本数为：")
    print(w2.shape[0])
    print("第三类的聚类样本数为：")
    print(w3.shape[0])
    plt.show()