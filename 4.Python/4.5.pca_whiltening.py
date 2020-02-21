# /usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt




def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #inputs是经过归一化处理的，所以这边就相当于计算协方差矩阵
    U,S,V = np.linalg.svd(sigma) #奇异分解
    epsilon = 0.1                #白化的时候，防止除数为0
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #计算zca白化矩阵
    return np.dot(ZCAMatrix, inputs)   #白化变换



digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

print("digits_train size")
# (3823, 65)
print(digits_train.shape)
# (1797, 65)
print("digits_test size")
print(digits_test.shape)


X_digits = digits_train[np.arange(64)] #0--64共有65维特征
y_digits = digits_train[64]

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)
print(type(X_pca))

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        # FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
        #   px = X_pca[:, 0][y_digits.as_matrix() == i]

        px = X_pca[:, 0][y_digits.values == i]
        py = X_pca[:, 1][y_digits.values == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
plot_pca_scatter()