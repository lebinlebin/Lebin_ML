#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
SVM分类鸢尾花数据
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

if __name__ == "__main__":
    iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
    path = '/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/9.Regression/iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x,y = data[range(4)],data[4]#0,1,2,3列为x,4为y
    print(y)
    #这里选择0，1是为了画图方便
    x, y = data[[0, 1]], pd.Categorical(data[4]).codes  #将类别标记为0，1，2的数字

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    print("x_train:   ",x_train)
    print("y_train:   ",y_train)

    # 分类器
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    """
    numpy的ravel() 和 flatten()函数
    首先声明两者所要实现的功能是一致的（将多维数组降位一维）。
    这点从两个单词的意也可以看出来，ravel(散开，解开)，flatten（变平）。
    两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，
    对拷贝所做的修改不会影响（reflects）原始矩阵，
    而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），
    会影响（reflects）原始矩阵。
    """
    # 训练集准确率  这里clf.score其实就是训练集的准确率
    print(clf.score(x_train, y_train))  # 精度
    # Return the mean accuracy on the given test data and labels.
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))


    #测试集准确率
    print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))

    # decision_function
    print("x_train[:5]")
    print(x_train[:5])#输出前五个样本
    #decision_function返回三个数  计算每个样本到三个分类器(0类，1类，2类)决策函数平面的距离，谁大就属于哪个类别
    print('decision_function:\n', clf.decision_function(x_train))
    #Evaluates the decision function for the samples in X.
    print('\npredict:\n', clf.predict(x_train))



    # 画图
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    grid_hat = clf.predict(grid_test)       # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
    plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('鸢尾花SVM二特征分类', fontsize=16)
    plt.grid(b=True, ls=':')
    plt.tight_layout(pad=1.5)
    plt.show()
