#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
验证QR分解
求特征值
求矩阵的逆
"""
from scipy.linalg import orth
from numpy.linalg import  matrix_rank
import numpy as np
import  math

def is_same(a,b):
    n = len(a)
    for i in range(n):
        if(math.fabs(a[i]-b[i]) > 1e-6):
            return False
    return  True

if __name__ == "__main__":
    """
    该数据来自于课上例子，人分三等级的状态转移模型
    """
    a = np.array((0.65,0.28,0.07,0.15,0.67,0.18,0.12,0.36,0.52))
    n = int(math.sqrt(len(a)))
    print(n)
    a = a.reshape((n,n))
    value,v = np.linalg.eig(a)

    time = 0
    while (time == 0) or (not is_same(np.diag(a),v)):
        v = np.diag(a)
        q,r = np.linalg.qr(a)
        a = np.dot(r,q)
        time += 1
        print("正交矩阵：\n",q)
        print("三角矩阵：\n",r)
        print("近似矩阵：\n",a)
    print("次数：",time,"近似值：",np.diag(a))
    print("精确特征值：",value)


"""
迭代求矩阵求逆
"""
# def inverse(a):
#     b = np.zeros_like(a)
#     n = len(a)
#     c = np.eye(n)
#     alpa = 1
#     for times in range(200):
#         for i in range(n):
#             for j in range(n):
#                 err = c[i][j]-:
#                 for k in range(n):
#                     b[j][k] += a
#     return b.T


"""
直接调用
"""
kernel = np.array([1, 1, 1, 2]).reshape((2, 2))
print(kernel)
print(np.linalg.inv(kernel))