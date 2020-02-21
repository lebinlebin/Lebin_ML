#!/usr/bin/python
#  -*- coding:utf-8 -*-
"""
奇异值分解(Singular Value Decomposition)是一种重要的矩阵分解方法，可以看做对称方阵在任意矩阵上的推广。
 Singular：突出的、奇特的、非凡的
 似乎更应该称之为“优值分解”
   假设A是一个m× n阶实矩阵，则存在一个分解使得：
    A(mxn) = U(mxm)Σ(mxn) (V(nxn))^T
 通常将奇异值由大而小排列。这样， Σ便能由A唯一确定了。
   与特征值、特征向量的概念相对应：

 Σ对角线上的元素称为矩阵A的奇异值；
 U的第i列称为A的关于σi的左奇异向量；
 V的第i列称为A的关于σi的右奇异向量
"""
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
    a[a < 0] = 0
    a[a > 255] = 255
    # a = a.clip(0, 255)
    return np.rint(a).astype('uint8')


def restore2(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K+1):
        for i in range(m):
            a[i] += sigma[k] * u[i][k] * v[k]
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')


if __name__ == "__main__":
    A = Image.open("/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/5.Package/lena.png", 'r')
    print(A)
    output_path = r'.\SVD_Output'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    print(a.shape)
    K = 50
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])#通道R
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])#通道G
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])#通道B

    plt.figure(figsize=(11, 9), facecolor='w')
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    for k in range(1, K+1):
        print(k)
        R = restore1(sigma_r, u_r, v_r, k)
        G = restore1(sigma_g, u_g, v_g, k)
        B = restore1(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), axis=2)
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
        if k <= 12:
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')
            plt.title('奇异值个数：%d' % k)
    plt.suptitle('SVD与图像分解', fontsize=20)
    plt.tight_layout(0.3, rect=(0, 0, 1, 0.92))
    # plt.subplots_adjust(top=0.9)
    plt.show()
