#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy import special

"""
面试题：
福原爱与刘诗雯正在乒乓球比赛，若任何一球刘诗雯赢得概率是60%。则对于11分制的一局，刘诗雯
获胜的概率有多大？
为了计算简便，暂不考虑分差必须大于等于2
注：如果考虑分差大于等于2，结果相差非常小

0.825622133638/0.836435199842

如果考虑"五局三胜制"  或者 "七局四胜制" ，则刘诗雯获胜的概率有多大？？
0.966274558546
0.983505058096

"""
if __name__ == '__main__':
    method = 'simulation'
    # simulation
    # 1.暴力模拟
    if method == 'simulation':
        p = 0.6
        a, b, c = 0, 0, 0
        t, T = 0, 1000000
        while t < T:
            a = b = 0 #初始分数都为0
            while (a <= 11) and (b <= 11):
                if np.random.uniform() < p:
                    a += 1#刘诗雯加1分
                else:
                    b += 1#福原爱加一分
            if a > b:
                c += 1#一局比赛结束，刘诗雯分数大于福原爱，则c+1
            t += 1#下一局，一直持续T局
        print(float(c) / float(T))#刘诗雯赢的概率

    # 2.直接计算  利用负二项分布计算。对于一系列成败实验 ，每次成功的概率为p,持续实验指导r次成功（r为正整数），则总实验次数X的概率服从负二项分布
    # 不考虑差2分获胜
    elif method == 'simple':
        answer = 0
        p = 0.6     # 每分的胜率
        N = 3      # 每局多少分
        for x in np.arange(N):  # x为对手得分
            answer += special.comb(N + x - 1, x) * ((1-p) ** x) * (p ** N)
        print(answer)
    # 3.严格计算
    else:
        answer = 0
        p = 0.525  # 每分的胜率
        N = 1000  # 每局多少分
        for x in np.arange(N-1):  # x为对手得分：11:9  11:8  11:7  11:6...
            answer += special.comb(N + x - 1, x) * ((1 - p) ** x) * (p ** N)
            print(x, answer)
        p10 = special.comb(2*(N-1), N-1) * ((1-p)*p) ** (N-1)   # 10:10的概率
        t = 0
        for n in np.arange(100):    # {XO}(0,)|OO   思考：可以如何简化？
            t += (2*p*(1-p)) ** n * p * p
        answer += p10 * t
        print(answer)
