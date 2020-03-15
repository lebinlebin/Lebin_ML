# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
密度聚类方法
 密度聚类方法的指导思想是，只要样本点的密度大于某阈值，则将该样本添加到最近的簇中。
 这类算法能克服基于距离的算法只能发现“类圆形”(凸)的聚类的缺点，可发现任意形状的聚类，
且对噪声数据不敏感。但计算密度单元的计算复杂度大，需要建立空间索引来降低计算量。
 DBSCAN
 密度最大值算法


DBSCAN算法
 DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
 一个比较有代表性的基于密度的聚类算法。与划分和层次聚类方法不同，它将簇定义为密度相连的点的最大集合，
能够把具有足够高密度的区域划分为簇，并可在有“噪声”的数据中发现任意形状的聚类。

DBSCAN算法的若干概念
 对象的ε-邻域：给定对象在半径ε内的区域。
 核心对象：对于给定的数目m，如果一个对象的ε-核心对象对于给定的数目m如果个对象的ε
 直接密度可达：给定一个对象集合D，如果p是在q直接密度可达给定个对象集合D如果p是在q的ε邻域内而q是个核心对
 如图ε=1cm，m=5，q是一个核心对象，从对象q出发到对象p是直接密度可达的。

DBSCAN算法的若干概念
 密度可达：如果存在一个对象链p1p2…pn，p1=q，pn=p，对密度可达如果存在个对象链p1p2pn，p1q，pnp，对pi∈D，(1≤i ≤n)，pi+1是从pi关于ε和m
 密度相连：如果对象集合D中存在一个对象o，使得对象p和q密度相连如果对象集合D中存在个对象o，使得对象p和q是从o关于ε和m密度可达的，那么对象p和q是关于ε和m密度 相连的。
 簇：一个基于密度的簇是最大的密度相连对象的集合。
 噪声：不包含在任何簇中的对象称为噪声。
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)
    data = StandardScaler().fit_transform(data)
    # 数据1的参数：(epsilon, min_sample)
    params = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))

    # 数据2
    # t = np.arange(0, 2*np.pi, 0.1)
    # data1 = np.vstack((np.cos(t), np.sin(t))).T
    # data2 = np.vstack((2*np.cos(t), 2*np.sin(t))).T
    # data3 = np.vstack((3*np.cos(t), 3*np.sin(t))).T
    # data = np.vstack((data1, data2, data3))
    # # # 数据2的参数：(epsilon, min_sample)
    # params = ((0.5, 3), (0.5, 5), (0.5, 10), (1., 3), (1., 10), (1., 20))

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(9, 7), facecolor='w')
    plt.suptitle('DBSCAN聚类', fontsize=15)

    for i in range(6):
        eps, min_samples = params[i]
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(data)
        y_hat = model.labels_

        core_indices = np.zeros_like(y_hat, dtype=bool)
        core_indices[model.core_sample_indices_] = True

        y_unique = np.unique(y_hat)
        n_clusters = y_unique.size - (1 if -1 in y_hat else 0)
        print(y_unique, '聚类簇的个数为：', n_clusters)

        # clrs = []
        # for c in np.linspace(16711680, 255, y_unique.size):
        #     clrs.append('#%06x' % c)
        plt.subplot(2, 3, i+1)
        clrs = plt.cm.Spectral(np.linspace(0, 0.8, y_unique.size))
        print(clrs)
        for k, clr in zip(y_unique, clrs):
            cur = (y_hat == k)
            if k == -1:
                plt.scatter(data[cur, 0], data[cur, 1], s=10, c='k')
                continue
            plt.scatter(data[cur, 0], data[cur, 1], s=15, c=clr, edgecolors='k')
            plt.scatter(data[cur & core_indices][:, 0], data[cur & core_indices][:, 1], s=30, c=clr, marker='o', edgecolors='k')
        x1_min, x2_min = np.min(data, axis=0)
        x1_max, x2_max = np.max(data, axis=0)
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.plot()
        plt.grid(b=True, ls=':', color='#606060')
        plt.title(r'$\epsilon$ = %.1f  m = %d，聚类数目：%d' % (eps, min_samples, n_clusters), fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
