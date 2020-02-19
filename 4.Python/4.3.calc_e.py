#!/usr/bin/python
#  -*- coding:utf-8 -*-
"""
给定正实数x，计算e^x
任何一个小数x，可以写成一个自然数+小数(余项)的形式
x=N+alpa  其中 alpa <= 0.5
另 x=k*ln2 +r ,  |r|<=0.5*ln2
  =>  e^x = e^(k*ln2 + r)
          = e^k*ln2 * e^r
          = 2^k * e^r   其中k和r已知
"""
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


def calc_e_small(x):
    n = 10
    f = np.arange(1, n+1).cumprod()
    b = np.array([x]*n).cumprod()
    return np.sum(b / f) + 1


def calc_e(x):
    reverse = False
    if x < 0:   # 处理负数
        x = -x
        reverse = True
    ln2 = 0.69314718055994530941723212145818
    c = x / ln2
    a = int(c+0.5)
    b = x - a*ln2
    y = (2 ** a) * calc_e_small(b)
    if reverse:
        return 1/y
    return y


if __name__ == "__main__":
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 4, 20)
    t = np.concatenate((t1, t2))
    print(t)     # 横轴数据
    y = np.empty_like(t)
    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^', x, ' = ', y[i], '(近似值)\t', math.exp(x), '(真实值)')
    plt.figure(facecolor='w')
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2)
    plt.title('Taylor展式的应用 - 指数函数', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True, ls=':')
    plt.show()
"""
pandas.Series.cumprod 官方文档
cumprod()累积连乘

Series.cumprod(axis=None, skipna=True, *args, **kwargs)
#实现功能：Return cumulative product over a DataFrame or Series axis.
#实现功能：Returns a DataFrame or Series of the same size containing the cumulative product.
#return：scalar or Series
cumsum()累积连加

pandas.Series.prod官方文档
Series.prod(axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs)
# 实现功能：Return the product of the values for the requested axis.
# return：scalar or Series
优点没看明白，因为常规情况下，所用的.prod()并非pandas下的函数，而是numpy下的函数。

numpy.prod官方文档
numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<class numpy._globals._NoValue>)
# 实现功能：Return the product of array elements over a given axis.
# return：product_along_axis : ndarray
返回给定轴上数组元素的乘积。

跟cumprod不同，cumprod是计算当前一个累积乘上前面所有的数据，更多是一个list；prod返回的是给定这个轴上最终一个值。

"""