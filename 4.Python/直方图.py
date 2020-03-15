# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
docnames=['X1','X2','TX3','X4','X5']
doctopic=[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.09,0.11,0.09,0.11,0.09,0.11,0.09,0.11,0.09,0.11],[0.12,0.08,0.12,0.08,0.12,0.08,0.12,0.08,0.12,0.08],[0.14,0.06,0.14,0.06,0.14,0.06,0.14,0.06,0.14,0.06],[0.15,0.05,0.15,0.05,0.15,0.05,0.15,0.05,0.15,0.05]]
doctopic=np.array(doctopic)
N, K = doctopic.shape
ind = np.arange(N)
width = 0.5
plots = []
height_cumulative = np.zeros(N)
for k in range(K):
	color = plt.cm.coolwarm(k/K, 1)
	if k == 0:
		p = plt.bar(ind, doctopic[:, k], width, color=color)
	else:
		p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
	height_cumulative += doctopic[:, k]
	plots.append(p)
plt.ylim((0, 1))
plt.ylabel('Probability')
plt.title('SAMPLE 1')
plt.xticks(ind, docnames)   #改变X轴示例位置
plt.yticks(np.arange(0, 0.1, 1))  #改变Y轴示例位置
topic_labels = ['X{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels,loc='center right')
plt.show()