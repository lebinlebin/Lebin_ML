# /usr/bin/python
# -*- coding:utf-8 -*-
#20.2.MarkovModel.py
"""
寻找马航MH370
2014年3月8日，马来西亚航空公司MH370航班(波
2014年3月8日马来西亚航空公司MH370航班(波
音777200ER)客机凌晨0:41分从吉隆坡飞往北京
凌晨1:19分马航MH370与空管失去联系凌晨
失。
 2015年7月29日在法属留尼汪岛(l‘île de la Reunion)
2015年7月29日在法属留尼汪岛(l‘île de la Reunion)
发现襟副翼残骸2015年8月6日马来西亚宣布
该残骸确属马航MH370随后法国谨慎宣布“有
很强的理由推测认为残骸属于马航MH370航班
技术验证加以确认。”

MH370最后消失区域
 可否根据雷达最后
消失区域和洋流、
大气等因素：
 判断留尼汪岛是否
位于可能区域？
 残骸漂流到该岛屿
的概率有多大？

马尔科夫模型模拟实验
 概率优势方向：Direct/Sin/Random
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib import animation
from PIL import Image


def update(f):
    global loc
    if f == 0:
        loc = loc_prime
    next_loc = np.zeros((m, n), dtype=np.float)
    for i in np.arange(m):
        for j in np.arange(n):
            next_loc[i, j] = calc_next_loc(np.array([i, j]), loc, directions)
    loc = next_loc / np.max(next_loc)
    im.set_array(loc)

    # Save
    if save_image:
        if f % 3 == 0:
            image_data = plt.cm.coolwarm(loc) * 255
            image_data, _ = np.split(image_data, (-1, ), axis=2)
            image_data = image_data.astype(np.uint8).clip(0, 255)
            output = '/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/20.BayesianNetwork/Pic2/'
            if not os.path.exists(output):
                os.mkdir(output)
            a = Image.fromarray(image_data, mode='RGB')
            a.save('%s%d.png' % (output, f))
    return [im]


def calc_next_loc(now, loc, directions):
    near_index = np.array([(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)])
    directions_index = np.array([7, 6, 5, 0, 4, 1, 2, 3])
    nn = now + near_index
    ii, jj = nn[:, 0], nn[:, 1]
    ii[ii >= m] = 0
    jj[jj >= n] = 0
    return np.dot(loc[ii, jj], directions[ii, jj, directions_index])


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=300, edgeitems=8)
    np.random.seed(0)

    save_image = True
    style = 'Random'   # Sin/Direct/Random
    m, n = 50, 100
    directions = np.random.rand(m, n, 8)

    if style == 'Direct':
        directions[:,:,1] = 10
    elif style == 'Sin':
        x = np.arange(n)
        y_d = np.cos(6*np.pi*x/n)
        theta = np.empty_like(x, dtype=np.int)
        theta[y_d > 0.5] = 1
        theta[~(y_d > 0.5) & (y_d > -0.5)] = 0
        theta[~(y_d > -0.5)] = 7
        directions[:, x.astype(np.int), theta] = 10
    directions[:, :] /= np.sum(directions[:, :])
    print (directions)

    loc = np.zeros((m, n), dtype=np.float)
    loc[int(m/2), int(n/2)] = 1
    loc_prime = np.empty_like(loc)
    loc_prime = loc
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    im = plt.imshow(loc/np.max(loc), cmap='coolwarm')
    anim = animation.FuncAnimation(fig, update, frames=300, interval=50, blit=True)#interval=50 50ms刷新一次
    plt.tight_layout(1.5)
    plt.show()
