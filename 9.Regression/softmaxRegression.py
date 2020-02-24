# -*- coding:utf-8 -*-
"""
SoftMax回归python代码实现
"""

from numpy import *
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def loadData():             # 数据读入处理
    data = pd.read_csv("iris.data", header=None)
    m, n = shape(data)
    y = pd.Categorical(data[4]).codes        # 对不同的标签进行编码
    x_prime = mat(data[range(n - 1)])        # 将所有的样本特征复制给x_prime,   最后一列是标签
    x = mat(ones((m, n)))                    # x0 是1，其余列是特征
    for i in range(n-2):
        x[:, i + 1] = x_prime[:, i]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=0)  # 训练集和测试集划分
    return x_train, x_test, y_train, y_test

def Cal_OnePredect(thetaI,x):       # 计算概率模型中的分子项
    return exp(thetaI*x.T)

def cal_Predect(theta,x,K,k):                                # 有概率模型计算发生的概率
    Numerator = Cal_OnePredect(theta[k],x)                     # 分子项
    Denominator = sum(Cal_OnePredect(theta[i],x) for i in range(K))  # 分母项
    return Numerator/Denominator

def Cal_Theta(x_train,y_train):           # x_train 已经包括了 x0 = 1 项
    m,n = shape(x_train)                  # m 样本数目， n  = 特征数目 + 1
    K = len(set(y_train))                 # 类别数目
    theta = mat(zeros((K,n)))             # theta 是 K 行 n 列的矩阵
    alpha = 0.001
    weight_lambda = 0.01
    for i in range(500):               # 设置迭代次数
        for j in range(m):                # 对每个样本
            for k in range(K):
                y_trun = int(y_train[j]==k)         # 是否属于当前类别
                Predect = cal_Predect(theta,x_train[j],K,k)
                theta[k] = theta[k] + alpha*((y_trun - Predect)*x_train[j] - weight_lambda*theta[k])
    return theta


def ModelPredect(theta,x_test):          # 预测函数, 返回预测的结果
    labels = []
    for x in (x_test):
        result = theta * x.T
        m = argmax(result)
        labels.append(m)
    return labels

def Accuracy_Score(Text,Predect):        # 计算正确率
    m = len(Text)
    j = 0.0
    for i in range(m):
        if Text[i] == Predect[i]:
            j += 1.0
    return j/m

if __name__=="__main__":
    x_train, x_test, y_train, y_test = loadData()
    # y_train 数据类型 numpy.ndarray, 类别标签
    # 输入：x_train 数据类型 m x n 的 Matrix， m = 样本数目， n = 特征数目 + 1
    theta = Cal_Theta(x_train, y_train)
    print ("theta =\n", theta)               # theta 是 k 行 n 的矩阵，k 是类别数目，n 是特征数目加1

    # 训练集上的预测结果
    y_train_pred = ModelPredect(theta,x_train)
    acc_Train = Accuracy_Score(y_train,y_train_pred)
    print ('\t训练集准确率: %.4f%%' % (100 * acc_Train))
    #测试集上的预测结果
    y_test_pred = ModelPredect(theta,x_test)
    acc_Test = Accuracy_Score(y_test,y_test_pred)
    print ('\t测试集准确率: %.4f%%\n' % (100 * acc_Test))