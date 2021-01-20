import numpy as np
import os
import skimage.io
import skimage.color
from sklearn import svm,datasets
import matplotlib.pyplot as plt
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
'''
data_dir = 'dataset/'#文件地址/名称
classes = os.listdir(data_dir)
data = []
all_image = np.zeros(100*100)
#all_image = np.array(all_image)
all_label = []
for cls in classes:
    files = os.listdir(data_dir+cls)
    for f in files:        
        img = skimage.io.imread(data_dir+cls+"/"+f)
        img = skimage.color.rgb2gray(img)#将图片转为灰度图
        img = img.reshape(1,100*100)
        #print(img)
        all_image = np.vstack((all_image,img))
        all_label.append(cls)
        
       

all_image = np.array(all_image)
all_label = np.array(all_label)

all_label = all_label.reshape(len(all_label),1)

all_image = np.delete(all_image,0,0) #删除第一行
all_data = np.hstack((all_label,all_image))

print(type(all_image[0,0]))
print(all_data)
'''

data_dir = 'dataset/'#文件地址/名称
classes = os.listdir(data_dir)
data = []
all_image = np.zeros(100*100)
#all_image = np.array(all_image)
all_label = []
for cls in classes:
    files = os.listdir(data_dir+cls)
    for f in files:        
        img = skimage.io.imread(data_dir+cls+"/"+f)
        img = skimage.color.rgb2gray(img)#将图片转为灰度图
        img = img.reshape(1,100*100)
        #print(img)
        all_image = np.vstack((all_image,img))
        all_label.append(int(cls))
        
       

all_image = np.array(all_image)
all_label = np.array(all_label)

all_label = all_label.reshape(len(all_label),1)

all_image = np.delete(all_image,0,0) #删除第一行
all_image.dtype = 'float32'

pca = decomposition.PCA(n_components=10, svd_solver='auto',
          whiten=True).fit(all_image)
# PCA降维后的总数据集
all_image = pca.transform(all_image)
all_data = np.hstack((all_label,all_image))

#print(all_image)
#print(all_label)      
#print(all_data)

all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = []  # 总数据对应的类标签
all_data_set = all_data[ : ,1: ]
all_data_label = all_data[:,0]

X = np.array(all_data_set)
y = np.array(all_data_label)

n_estimators = [10,11]
criterion_test_names = ["gini", "entropy"]#测试 系数与熵
                                          #分类树: 基尼系数 最小的准则 在sklearn中可以选择划分的默认原则                                      
                                          #决策树：criterion:默认是’gini’系数，也可以选择信息增益的熵’entropy



def RandomForest(n_estimators,criterion):
    # 十折交叉验证计算出平均准确率
    # 交叉验证，随机取
    kf = KFold(n_splits=10, shuffle=True)
    precision_average = 0.0
    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}  # 自动穷举出最优的C参数
    # clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced', gamma=param),
    #                    param_grid)s

    parameter_space = {
        "min_samples_leaf": [2, 4, 6], }#参数空间   随机森林RandomForest使用了CART决策树作为弱学习器
    clf = GridSearchCV(RandomForestClassifier(random_state=14,n_estimators = n_estimators,criterion = criterion), parameter_space, cv=5)
    for train, test in kf.split(X):
        clf = clf.fit(X[train], y[train].astype('int'))
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


for criterion in criterion_test_names:
    print("criterion")
    print(criterion)
    x_label = []
    y_label = []
    for i in range(10,15):
        print(i)
        y_label.append(RandomForest(i,criterion))
        x_label.append(i)
    plt.plot(x_label, y_label, label=criterion)
# print("done in %0.3fs" % (time() - t0))
plt.xlabel("criterion")
plt.ylabel("Precision")
plt.title('Different  Contrust')
plt.legend()
plt.show()