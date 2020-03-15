# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

docnames = ['Austen_Emma', 'Austen_Pride', 'Austen_Sense', 'CBronte_Jane', 'CBronte_Professor', 'CBronte_Villette']
doctopic = [[0.0625, 0.1736, 0.0819, 0.4649, 0.2171], [0.0574, 0.1743, 0.0835, 0.4008, 0.2839],
            [0.0599, 0.1645, 0.0922, 0.2034, 0.4801], [0.189, 0.1897, 0.3701, 0.1149, 0.1362],
            [0.2772, 0.2681, 0.2387, 0.0838, 0.1322], [0.3553, 0.193, 0.2409, 0.0865, 0.1243]]

xmajorLocator = MultipleLocator(1)  # 将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%d')  # 设置x轴标签文本的格式
xminorLocator = MultipleLocator(10)  # 将x轴次刻度标签设置为5的倍数

doctopic = np.array(doctopic)

doctopic = doctopic.transpose()

N, K = doctopic.shape
# topic_labels = ['No #{}'.format(k) for k in range(K)]
# topic_labels=['0','30','60','90','120','150','180','210','240']

plt.pcolor(doctopic, norm=None, cmap='Blues')  # Blues
plt.yticks(np.arange(doctopic.shape[0]) + 0.5, docnames)
plt.xticks(np.arange(doctopic.shape[1]))  # topic_labels,rotation=0
plt.gca().invert_yaxis()
plt.xticks(rotation=90)
plt.colorbar(cmap='Blues')  # Oranges,Purples,Reds,hot
plt.tight_layout()

ax = plt.subplot(111)
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)

plt.show()