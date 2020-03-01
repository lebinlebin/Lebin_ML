"""
使用随机森林进行特征选择
一、特征选择

在我们做特征工程时，当我们提取完特征后，可能存在并不是所有的特征都能分类起到作用的问题，
这个时候就需要使用特征选择的方法选出相对重要的特征用于构建分类器。
此外，使用特征选择这一步骤也大大减少了训练的时间，而且模型的拟合能力也不会出现很大的降低问题。
在特征选择的许多方法中，我们可以使用随机森林模型中的特征重要属性来筛选特征，
并得到其与分类的相关性。由于随机森林存在的固有随机性，该模型可能每次给予特征不同的重要性权重。
但是通过多次训练该模型，即每次通过选取一定量的特征与上次特征中的交集进行保留，以此循环一定次数，
从而我们最后可以得到一定量对分类任务的影响有重要贡献的特征。

"""
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt

with open('training_df.pkl', 'rb') as f:
    df = pickle.load(f)
print("data loaded")

y = df["y"]  # 获取标签列
X = df.drop("y", axis=1)  # 剩下的所有特征

for i in range(10):  # 这里我们进行十次循环取交集
    tmp = set()
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(X, y)
    print("training finished")

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]  # 降序排列
    for f in range(X.shape[1]):
        if f < 50:  # 选出前50个重要的特征
            tmp.add(X.columns[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))

    selected_feat_names = tmp
    print(len(selected_feat_names), "features are selected")

plt.title("Feature Importance")
plt.bar(range(X.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X.shape[1]),
           X.columns[indices],
           rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

with open(r'selected_feat_names.pkl', 'wb') as f:
    pickle.dump(list(selected_feat_names), f)