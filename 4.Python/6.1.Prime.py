#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
1、 计算素数
2、赔率
赔率最早出现在赛马中,1790年由英国人奥格登发明。
 中国从2001年发行足彩开始引入赔率。
赔率的举例定义：
浔阳江畔艄公张横和张顺正进行400米自由泳比赛，
宋江开赌场做庄，规定： 张横赢赔率为3， 张顺赢
赔率为2。假定不存在平局。赌徒李逵为张横下注10两。
 比赛结束后，若最终张横赢，则宋江付赌徒李逵30两
(10× 3)，赌资10两归庄家宋江所有，即李逵赚20两。
 若张顺赢，赌资10两归庄家宋江所有，即李逵赔10两。


计算赔率
 拼团人数当时是1026人，尚有两天结束，根
据历史先验，假定1天参团人数为100人，则
最终参团人数为1226左右。考虑到3月12日
为星期日，参团人数或许略低，因此大体参
团区间可能是[1180,1230]。
计算该区间的素数
 [1181 1187 1193 1201 1213 1217 1223 1229]
 素数的概率： 0.157 公正赔率： 6.375
 合数的概率： 0.843 公正赔率： 1.186
最终拼团人数是素数还是合数作为押注
"""
import numpy as np
from time import time
import math


def is_prime(x):
    return 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)]


def is_prime3(x):
    flag = True
    for p in p_list2:
        if p > math.sqrt(x):
            break
        if x % p == 0:
            flag = False
            break
    if flag:
        p_list2.append(x)
    return flag


if __name__ == "__main__":
    a = 2
    b = 1000
    """
    计算2到1000的所有素数
    """

    # 方法1：直接计算  要想判断p是不是素数就看从2，3，4，5，6，...一直到根号p，看看有没有能够整除p的就可以
    t = time()                                                        #range左闭右开的所以要+1
    p = [p for p in range(a, b) if 0 not in [p % d for d in range(2, int(math.sqrt(p)) + 1)]]
    print("----------------方法1： 直接计算 计算耗时--------------")
    print(time() - t)
    print(p)

    # 方法2：利用filter
    t = time()
    p = list(filter(is_prime, list(range(a, b))))
    print("----------------方法2： 利用filter 计算耗时--------------")
    print(time() - t)
    print(p)
    # 利用filter会时间会快一些

    # 方法3：利用filter和lambda
    t = time()
    is_prime2 = (lambda x: 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)])
    p = list(filter(is_prime2, list(range(a, b))))
    print("----------------方法3： 利用filter和lambda 计算耗时--------------")
    print(time() - t)
    print(p)

    # 方法4：定义
    t = time()
    p_list = []
    for i in range(2, b):  #计算2到b的所有素数
        flag = True
        for p in p_list:
            if p > math.sqrt(i):
                break
            if i % p == 0:
                flag = False#合数
                break
        if flag:  #是素数
            p_list.append(i)
    print("----------------方法4：利用定义 计算耗时--------------")
    print(time() - t)
    print(p_list)

    # 方法5：定义和filter
    p_list2 = []
    t = time()
    list(filter(is_prime3, list(range(2, b))))
    print("----------------方法5：定义和filter 计算耗时--------------")
    print(time() - t)
    print(p_list2)

    print('---------------------')
    a = 750
    b = 900
    p_list2 = []
    np.set_printoptions(linewidth=150)
    p = np.array(list(filter(is_prime3, list(range(2, b+1)))))
    p = p[p >= a]
    print(p)
    p_rate = float(len(p)) / float(b-a+1)
    print('素数的概率：', p_rate, end='\t  ')
    print('公正赔率：', 1/p_rate)
    print('合数的概率：', 1-p_rate, end='\t  ')
    print('公正赔率：', 1 / (1-p_rate))

    alpha1 = 5.5 * p_rate
    alpha2 = 1.1 * (1 - p_rate)
    print('赔率系数：', alpha1, alpha2)
    print(1 - (alpha1 + alpha2) / 2)
    print((1 - alpha1) * p_rate + (1 - alpha2) * (1 - p_rate))
