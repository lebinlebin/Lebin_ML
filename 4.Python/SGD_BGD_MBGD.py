#coding=utf-8
import numpy as np
import random
# coding=utf-8
'''
随机梯度下降
'''
import numpy as np

# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
target_data = 3 * x + 8 + np.random.randn(m)

max_iter = 10000  # 最大迭代次数
epsilon = 1e-5

# 初始化权值
w = np.random.randn(2)
# w = np.zeros(2)

alpha = 0.001  # 步长
diff = 0.
error = np.zeros(2)
count = 0  # 循环次数

print ('随机梯度下降算法'.center(60, '='))

while count < max_iter:
    count += 1
    for j in range(m):
        diff = np.dot(w, input_data[j]) - target_data[j]  # 训练集代入,计算误差值
        # 这里的随机性表现在：一个样本更新一次参数！
        w = w - alpha * diff * input_data[j]

    if np.linalg.norm(w - error) < epsilon:  # 直接通过np.linalg包求两个向量的范数
        break
    else:
        error = w
print ('loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1]))



# coding=utf-8
"""
批量梯度下降
"""
import numpy as np
print ('批量梯度下降'.center(60, '='))

# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
target_data = 3 * x + 8 + np.random.randn(m)

# 停止条件
max_iter = 10000
epsilon = 1e-5

# 初始化权值
w = np.random.randn(2)
# w = np.zeros(2)

alpha = 0.001  # 步长
diff = 0.
error = np.zeros(2)
count = 0  # 循环次数

while count < max_iter:
    count += 1

    sum_m = np.zeros(2)

    for i in range(m):
        dif = (np.dot(w, input_data[i]) - target_data[i]) * input_data[i]
        sum_m = sum_m + dif
    '''
    for j in range(m):
        diff = np.dot(w, input_data[j]) - target_data[j]  # 训练集代入,计算误差值
        w = w - alpha * diff * input_data[j]
    '''
    w = w - alpha * sum_m

    if np.linalg.norm(w - error) < epsilon:
        break
    else:
        error = w
print ('loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1]))





# coding=utf-8
"""
小批量梯度下降
"""
import numpy as np
import random
print ('批量梯小批量梯度下降'.center(60, '='))

# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
target_data = 3 * x + 8 + np.random.randn(m)

# 两种终止条件
max_iter = 10000
epsilon = 1e-5

# 初始化权值
np.random.seed(0)
w = np.random.randn(2)
# w = np.zeros(2)

alpha = 0.001  # 步长
diff = 0.
error = np.zeros(2)
count = 0  # 循环次数

while count < max_iter:
    count += 1

    sum_m = np.zeros(2)
    index = random.sample(range(m), int(np.ceil(m * 0.2)))
    sample_data = input_data[index]
    sample_target = target_data[index]

    for i in range(len(sample_data)):
        dif = (np.dot(w, input_data[i]) - target_data[i]) * input_data[i]
        sum_m = sum_m + dif

    w = w - alpha * sum_m

    if np.linalg.norm(w - error) < epsilon:
        break
    else:
        error = w
print ('loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1]))