# -*- coding:utf-8 -*-
# /usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import matplotlib
from matplotlib.font_manager import _rebuild

"""


"""
def func(a):
    if a < 1e-6:
        return 0
    last = a
    c = a / 2
    while math.fabs(c - last) > 1e-6:
        last = c
        c = (c + a/c) / 2
    return c


if __name__ == '__main__':
    _rebuild()  # reload一下
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    x = np.linspace(0, 30, num=50)
    """
    numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    产生从start到stop的等差数列，num为元素个数，默认50个
    """
    func_ = np.frompyfunc(func, 1, 1)
    y = func_(x)
    # y = np.sqrt(x)
    plt.figure(figsize=(10, 5), facecolor='w')

    plt.plot(x, y, 'ro-', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.title('这段代码在计算什么？', fontsize=18)
    plt.show()

"""
np.frompyfunc()函数的使用 Numpy | 函数向量化
原创kudou1994 最后发布于2019-07-02 01:56:40 阅读数 686  收藏
展开
python自定义函数并调用：

def func_x(a):
	y = a + 1
	return y

print(func_x(1)) # 调用
# 结果：
# 2

但是如果a不是一个数，而是一个向量/数组呢？这时候就需要借助numpy中的通用函数np.frompyfunc(func, 1, 1)
这里的参数func指的是你要使用的函数，第二个参数为func中的参数个数，第三个参数为func中的返回值的个数
假设A = [0, 1, 2]

def func_x(a):
	y = a + 1
	return y
A = [0, 1, 2]
func_ = np.frompyfunc(func_x, 1, 1)
y = func_(A)
print(y)
# 结果：
# [1 2 3]
"""