# 导入NumPy函数库，一般都是用这样的形式(包括别名np，几乎是约定俗成的)
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from scipy.optimize import leastsq
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import CubicSpline
import scipy as sp
import math
import seaborn
def residual(t, x, y):
    return y - (t[0] * x ** 2 + t[1] * x + t[2])


def residual2(t, x, y):
    print(t[0], t[1])
    return y - (t[0]*np.sin(t[1]*x) + t[2])


# x ** x        x > 0
# (-x) ** (-x)  x < 0
def f(x):
    y = np.ones_like(x)#生成一个和x形状相同的向量用1来填充
    i = x > 0#对于x中大于零的挑出来，正常计算x^x
    y[i] = np.power(x[i], x[i])
    i = x < 0#对于小于零的x挑出来，(-x)^(-x)
    y[i] = np.power(-x[i], -x[i])
    return y
if __name__ == "__main__":

    # 5.绘图
    # 5.1 绘制正态分布概率密度函数
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 修改标题上上乱码  FangSong/黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False  #修改坐标轴上的乱码
    mu = 0
    sigma = 1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 51)
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    print(x.shape)
    print('x = \n', x)
    print(y.shape)
    print('y = \n', y)
    plt.figure(facecolor='w')#绘图背景改为白色
    # plt.plot(x, y, 'ro-', linewidth=2)
    plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
    #  第一个曲线  x,y用'r-'  red实线绘制,  第二个曲线  用 'go' green的'o'圆圈来绘制；对于线条来讲用宽度为2的线，对于圈来讲按照大小为8来绘制
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.title(u'高斯分布函数', fontsize=18)  #
    plt.grid(True)#画曲线的格子
    plt.show()

    # 5.2 损失函数：Logistic损失(-1,1)/SVM Hinge损失/ 0/1损失
    plt.figure(figsize=(10, 8))#图像大小默认为英寸
    x = np.linspace(start=-2, stop=3, num=1001, dtype=np.float)
    y_logit = np.log(1 + np.exp(-x)) / math.log(2)
    y_boost = np.exp(-x)
    y_01 = x < 0
    y_hinge = 1.0 - x
    y_hinge[y_hinge < 0] = 0

    plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
    plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
    plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
    plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2)
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig('1.png')
    plt.show()

    # 5.3 x^x
    plt.figure(facecolor='w')
    x = np.linspace(-1.3, 1.3, 101)
    y = f(x)
    plt.plot(x, y, 'g-', label='x^x', linewidth=2)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

    # 5.4 胸型线
    x = np.arange(1, 0, -0.001)
    y = (-3 * x * np.log(x) + np.exp(-(40 * (x - 1 / np.e)) ** 4) / 25) / 2
    plt.figure(figsize=(5, 7), facecolor='w')
    plt.plot(y, x, 'r-', linewidth=2)
    plt.grid(True)
    plt.title(u'胸型线', fontsize=20)
    plt.savefig('breast.png')
    plt.show()

    # 5.5 心形线
    t = np.linspace(0, 2 * np.pi, 100)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    plt.plot(x, y, 'r-', linewidth=2)
    plt.grid(True)
    plt.show()

    # # 5.6 渐开线
    t = np.linspace(0, 50, num=1000)
    x = t * np.sin(t) + np.cos(t)
    y = np.sin(t) - t * np.cos(t)
    plt.plot(x, y, 'r-', linewidth=2)
    plt.grid()
    plt.show()

    # Bar采样sin曲线
    x = np.arange(0, 10, 0.1)
    y = np.sin(x)
    plt.bar(x, y, width=0.04, linewidth=0.2)
    plt.plot(x, y, 'r--', linewidth=2)
    plt.title(u'Sin曲线')
    plt.xticks(rotation=-60)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()

    # 6. 概率分布
    # 6.1 均匀分布
    print("验证中心极限定理,均匀分布的叠加为高斯分布")
    x = np.random.rand(10000)
    t = np.arange(len(x))
    plt.hist(x, 30, color='m', alpha=0.5, label=u'均匀分布') #alpha=0.5表示透不透明，半透明的  30个bin
    plt.plot(t, x, 'g.', label=u'均匀分布')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # 6.2 验证中心极限定理
    t = 1000
    a = np.zeros(10000) #做1w个样本，以0填充
    for i in range(t):#一共迭代1000次
        a += np.random.uniform(-5, 5, 10000)#从 -5 到5 做均匀分布
    # 均匀分布叠加多次得到的是正态分布
    a /= t
    plt.hist(a, bins=30, color='g', alpha=0.5, normed=True, label=u'均匀分布叠加')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # 6.21 其他分布的中心极限定理
    lamda = 10
    p = stats.poisson(lamda)
    y = p.rvs(size=1000)#r->random v->value  s->sample 采样1000个，获得1000个随机的poisson分布的值
    # y即为一个满足lamda为7的poisson分布1000个样本的采样值
    mx = 30
    r = (0, mx)#r取从0到30
    bins = r[1] - r[0]#30-0
    plt.figure(figsize=(15, 8), facecolor='w')
    plt.subplot(121)#做一个1行2列的图,现在画第一个图
    plt.hist(y, bins=bins, range=r, color='g', alpha=0.8, normed=True)
    t = np.arange(0, mx + 1)
    plt.plot(t, p.pmf(t), 'ro-', lw=2) #p.pmf(t) 真正的概率质量函数
    plt.grid(True) #两个图，一个是线，一个是直方图

    N = 1000
    M = 10000
    plt.subplot(122)
    a = np.zeros(M, dtype=np.float)
    p = stats.poisson(lamda)
    for i in np.arange(N):
        a += p.rvs(size=M) #泊松分布叠加多次之后
    a /= N#除以次数N

    plt.hist(a, bins=20, color='g', alpha=0.8, normed=True)
    plt.grid(b=True)
    plt.show()



    # 6.3 Poisson分布
    # 验证Poisson分布
    x = np.random.poisson(lam=5, size=10000)
    print("--验证Poisson分布----")
    print(x)
    pillar = 15
    a = plt.hist(x, bins=pillar, normed=True, range=[0, pillar], color='g', alpha=0.5)
    #返回是什么？？ 连个array，即取值为0的对应的概率，取值为1对应的概率，等等；对a加和肯定为1.
    """
    (array([0.00740074, 0.03520352, 0.07780778, 0.1440144 , 0.18081808,
       0.17581758, 0.14371437, 0.10971097, 0.06120612, 0.0350035 ,
       0.01660166, 0.00740074, 0.00280028, 0.00160016, 0.00090009]), array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15.]), <a list of 15 Patch objects>)
    1.0
    """
    plt.grid()
    plt.show()
    print(a)
    print(a[0].sum())

    # # 6.4 直方图的使用
    print("-------直方图的使用------")
    mu = 2
    sigma = 3
    data = mu + sigma * np.random.randn(1000)
    h = plt.hist(data, 30, normed=1, color='#FFFFA0')#FFFFA0的 FFFFA0对应RGB  FF=R  FF=G  A0=B
    x = h[1]
    y = norm.pdf(x, loc=mu, scale=sigma)#
    plt.plot(x, y, 'r-', x, y, 'ro', linewidth=2, markersize=4)
    plt.grid()
    plt.show()

    # # 6.5 插值
    rv = poisson(5)
    x1 = a[1]  #对应取值； a[0]对应概率; a[1]对应取值
    y1 = rv.pmf(x1)
    itp = BarycentricInterpolator(x1, y1)  # 重心插值
    x2 = np.linspace(x.min(), x.max(), 50)
    y2 = itp(x2)
    cs = sp.interpolate.CubicSpline(x1, y1)  # 三次样条插值
    plt.plot(x2, cs(x2), 'm--', linewidth=5, label='CubicSpine')  # 三次样条插值
    plt.plot(x2, y2, 'g-', linewidth=3, label='BarycentricInterpolator')  # 重心插值
    plt.plot(x1, y1, 'r-', linewidth=1, label='Actural Value')  # 原始值
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    # 6.6 Poisson分布
    size = 1000
    lamda = 5
    p = np.random.poisson(lam=lamda, size=size)
    plt.figure()
    plt.hist(p, bins=range(3 * lamda), histtype='bar', align='left', color='r', rwidth=0.8, normed=True)
    plt.grid(b=True, ls=':')
    # plt.xticks(range(0, 15, 2))
    plt.title('Numpy.random.poisson', fontsize=13)

    plt.figure()
    r = stats.poisson(mu=lamda)
    p = r.rvs(size=size)
    plt.hist(p, bins=range(3 * lamda), color='r', align='left', rwidth=0.8, normed=True)
    plt.grid(b=True, ls=':')
    plt.title('scipy.stats.poisson', fontsize=13)
    plt.show()

    # 7. 绘制三维图像
    x, y = np.ogrid[-3:3:7j, -3:3:7j]#从-3到3取7个
    """
    [[-3.]
     [-2.]
     [-1.]
     [ 0.]
     [ 1.]
     [ 2.]
     [ 3.]]
    [[-3. -2. -1.  0.  1.  2.  3.]]
    """
    x1, y1 = np.mgrid[-3:3:101j, -3:3:101j] #得到平面的坐标
    print("-------------绘制三维图像mgrid-----------------")
    print(x1)
    print(y1)
    print("-------------绘制三维图像meshgrid-----------------")
    u = np.linspace(-3, 3, 101)#
    x, y = np.meshgrid(u, u)
    print(x)
    print(y)

    # 以上两种方式结果是一样的。
    z1 = np.exp(-(x ** 2 + y ** 2) / 2) / math.sqrt(2 * math.pi)
    z = x * y * np.exp(-(x ** 2 + y ** 2) / 2) / math.sqrt(2 * math.pi)
    # z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0.1)  #
    ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap=cm.gist_heat, linewidth=0.5)
    #cmap标识color map选定颜色； rstride=3, cstride=3标识每五个取一个点
    plt.show()
    # cmaps = [('Perceptually Uniform Sequential',
    #           ['viridis', 'inferno', 'plasma', 'magma']),
    #          ('Sequential', ['Blues', 'BuGn', 'BuPu',
    #                          'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
    #                          'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
    #                          'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
    #          ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
    #                              'copper', 'gist_heat', 'gray', 'hot',
    #                              'pink', 'spring', 'summer', 'winter']),
    #          ('Diverging', ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
    #                         'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
    #                         'seismic']),
    #          ('Qualitative', ['Accent', 'Dark2', 'Paired', 'Pastel1',
    #                           'Pastel2', 'Set1', 'Set2', 'Set3']),
    #          ('Miscellaneous', ['gist_earth', 'terrain', 'ocean', 'gist_stern',
    #                             'brg', 'CMRmap', 'cubehelix',
    #                             'gnuplot', 'gnuplot2', 'gist_ncar',
    #                             'nipy_spectral', 'jet', 'rainbow',
    #                             'gist_rainbow', 'hsv', 'flag', 'prism'])]



    # 8.1 scipy
    # 线性回归例1
    x = np.linspace(-2, 2, 50)
    A, B, C = 2, 3, -1
    y = (A * x ** 2 + B * x + C) + np.random.rand(len(x)) * 0.75

    t = leastsq(residual, [0, 0, 0], args=(x, y))
    theta = t[0]
    print('真实值：', A, B, C)
    print('预测值：', theta)
    y_hat = theta[0] * x ** 2 + theta[1] * x + theta[2]
    plt.plot(x, y, 'r-', linewidth=2, label=u'Actual')
    plt.plot(x, y_hat, 'g-', linewidth=2, label=u'Predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # # 线性回归例2
    x = np.linspace(0, 5, 100)
    a = 5
    w = 1.5
    phi = -2
    y = a * np.sin(w * x) + phi + np.random.rand(len(x)) * 0.5

    t = leastsq(residual2, [3, 5, 1], args=(x, y))
    theta = t[0]
    print('真实值：', a, w, phi)
    print('预测值：', theta)
    y_hat = theta[0] * np.sin(theta[1] * x) + theta[2]
    plt.plot(x, y, 'r-', linewidth=2, label='Actual')
    plt.plot(x, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

    # # 8.2 使用scipy计算函数极值
    a = opt.fmin(f, 1)
    b = opt.fmin_cg(f, 1)
    c = opt.fmin_bfgs(f, 1)
    print(a, 1 / a, math.e)
    print(b)
    print(c)

    # marker	description
    # ”.”	point
    # ”,”	pixel
    # “o”	circle
    # “v”	triangle_down
    # “^”	triangle_up
    # “<”	triangle_left
    # “>”	triangle_right
    # “1”	tri_down
    # “2”	tri_up
    # “3”	tri_left
    # “4”	tri_right
    # “8”	octagon
    # “s”	square
    # “p”	pentagon
    # “*”	star
    # “h”	hexagon1
    # “H”	hexagon2
    # “+”	plus
    # “x”	x
    # “D”	diamond
    # “d”	thin_diamond
    # “|”	vline
    # “_”	hline
    # TICKLEFT	tickleft
    # TICKRIGHT	tickright
    # TICKUP	tickup
    # TICKDOWN	tickdown
    # CARETLEFT	caretleft
    # CARETRIGHT	caretright
    # CARETUP	caretup
    # CARETDOWN	caretdown