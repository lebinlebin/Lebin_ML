# !/usr/bin/python
# -*- coding:utf-8 -*-
"""

层次聚类方法
 层次聚类方法对给定的数据集进行层次的分解，直到某种条件满足为止。具体又可分为：
 凝聚的层次聚类：AGNES算法
 一种自底向上的策略，首先将每个对象作为一个簇，然后合并这些原子簇为越来越大的簇，直到某个终结条件被满足。
 分裂的层次聚类：DIANA算法
 采用自顶向下的策略，它首先将所有对象臵于一个簇中，然后逐渐细分为越来越小的簇，直到达到了某个终结条件。

层次聚类的不同合并策略
AGNES和DIANA算法
 AGNES (AGglomerative NESting)算法最初将每个对象作为一个簇，然后这些簇根据某些准则被一步步地合并。
两个簇间的距离由这两个不同簇中距离最近的数据点对的相似度来确定；聚类的合并过程反复进行直到所有的对象最终满足簇数目。
 DIANA (DIvisive ANAlysis)算法是上述过程的反过程，属于分裂的层次聚类，
首先将所有的对象初始化到一个簇中，然后根据一些原则(比如最大的欧式距离)，将该簇分类。直到到达用户指定的簇数
目或者两个簇之间的距离超过了某个阈值。

AGNES中簇间距离的不同定义
 最小距离
 两个集合中最近的两个样本的距离
 容易形成链状结构

 最大距离
 两个集合中最远的两个样本的距离complete
 若存在异常值则不稳定

 平均距离
 1、两个集合中样本间两两距离的平均值average
 2、两个集合中样本间两两距离的平方和ward
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import sklearn.datasets as ds
import warnings


def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=UserWarning)
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    n_clusters = 4
    N = 400
    data1, y1 = ds.make_blobs(n_samples=N, n_features=2, centers=((-1, 1), (1, 1), (1, -1), (-1, -1)),
                              cluster_std=(0.1, 0.2, 0.3, 0.4), random_state=0)
    data1 = np.array(data1)
    n_noise = int(0.1*N)
    r = np.random.rand(n_noise, 2)
    data_min1, data_min2 = np.min(data1, axis=0)
    data_max1, data_max2 = np.max(data1, axis=0)


    r[:, 0] = r[:, 0] * (data_max1-data_min1) + data_min1
    r[:, 1] = r[:, 1] * (data_max2-data_min2) + data_min2

    data1_noise = np.concatenate((data1, r), axis=0)
    y1_noise = np.concatenate((y1, [4]*n_noise))

    data2, y2 = ds.make_moons(n_samples=N, noise=.05)
    data2 = np.array(data2)
    n_noise = int(0.1 * N)
    r = np.random.rand(n_noise, 2)
    data_min1, data_min2 = np.min(data2, axis=0)
    data_max1, data_max2 = np.max(data2, axis=0)
    r[:, 0] = r[:, 0] * (data_max1 - data_min1) + data_min1
    r[:, 1] = r[:, 1] * (data_max2 - data_min2) + data_min2
    data2_noise = np.concatenate((data2, r), axis=0)
    y2_noise = np.concatenate((y2, [3] * n_noise))

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm = mpl.colors.ListedColormap(['r', 'g', 'b', 'm', 'c'])
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.cla()

    # 'prime'，  原始取样本件最小距离
    #"ward",  平方和距离度量
    # "complete",  最远点距离度量
    # "average" 取平均的度量方法
    linkages = ("ward", "complete", "average")
    for index, (n_clusters, data, y) in enumerate(((4, data1, y1), (4, data1_noise, y1_noise),
                                                   (2, data2, y2), (2, data2_noise, y2_noise))):
        plt.subplot(4, 4, 4*index+1)
        plt.scatter(data[:, 0], data[:, 1], c=y, s=12, edgecolors='k', cmap=cm)
        plt.title('Prime', fontsize=12)
        plt.grid(b=True, ls=':')
        data_min1, data_min2 = np.min(data, axis=0)
        data_max1, data_max2 = np.max(data, axis=0)
        plt.xlim(extend(data_min1, data_max1))
        plt.ylim(extend(data_min2, data_max2))

        connectivity = kneighbors_graph(data, n_neighbors=7, mode='distance', metric='minkowski', p=2, include_self=True)
        connectivity = 0.5 * (connectivity + connectivity.T)
        for i, linkage in enumerate(linkages):
            ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                         connectivity=connectivity, linkage=linkage)
            ac.fit(data)
            y = ac.labels_
            plt.subplot(4, 4, i+2+4*index)
            plt.scatter(data[:, 0], data[:, 1], c=y, s=12, edgecolors='k', cmap=cm)
            plt.title(linkage, fontsize=12)
            plt.grid(b=True, ls=':')
            plt.xlim(extend(data_min1, data_max1))
            plt.ylim(extend(data_min2, data_max2))
    plt.suptitle('层次聚类的不同合并策略', fontsize=15)
    plt.tight_layout(0.5, rect=(0, 0, 1, 0.95))
    plt.show()
