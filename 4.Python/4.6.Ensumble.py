#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
对于某二分类问题，若构造了9个正确率都是0.6的分类器，采用少数服从多数的原则进行最终分类，
则最终分类正确率为多少？
若构造99个分类器呢？

9个分类器分类正确情况加和即为最终的分类正确概率。
p(分类正确的概率) = c(9,1) 0.6^1*0.4^8 + c(9,2) 0.6^2*0.4^7+ c(9,3)0.6^3*0.4^6 + ...
"""
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import reduce


def c(n, k):
    return reduce(operator.mul, list(range(n-k+1, n+1))) / reduce(operator.mul, list(range(1, k+1)))


def bagging(n, p):
    s = 0
    for i in range(n // 2 + 1, n + 1):
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s


if __name__ == "__main__":
    n = 100
    x = np.arange(1, n, 2)
    y = np.empty_like(x, dtype=np.float)
    for i, t in enumerate(x):
        y[i] = bagging(t, 0.6)
        if t % 10 == 9:
            print(t, '次采样正确率：', y[i])
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(x, y, 'ro-', lw=2)
    plt.xlim(0, n)
    plt.ylim(0.6, 1)
    plt.xlabel('采样次数', fontsize=16)
    plt.ylabel('正确率', fontsize=16)
    plt.title('Bagging', fontsize=20)
    plt.grid(b=True)
    plt.show()
