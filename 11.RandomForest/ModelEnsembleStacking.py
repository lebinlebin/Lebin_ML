"""
Machine Learning】模型融合之Stacking
一、Stacking简介
  Stacking(stacked generalization)是在大数据竞赛中不可缺少的武器，
其指训练一个用于组合(combine)其他多个不同模型的模型，具体是说首先我们使用不同的算法或者其他方法能够训练出多个不同的模型，
然后将这些模型的输出作为新的数据集，即将这些训练的模型的输出再作为为输入训练一个模型，最后得到一个最终的输出，下图为Stacking的大致流程图：

如果可以选用任意的组合算法，那么理论上，Stacking可以表示上面提到的各种Ensemble方法。但是在实际应用中通常使用单层logistic回归作为组合模型。
二、代码示例
  在这里使用了mlxtend库，它可以很好地完成对sklearn模型地stacking。
"""
# -*- coding: utf-8 -*-

import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier

with open('../data/training_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(r'../data/selected_feat_names.pkl', 'rb') as f:
    selected_feat_names = pickle.load(f)
print("data loaded")

# train on full data set
y = df["attack_type"].values
X = df[selected_feat_names].values

xgb = XGBClassifier(learning_rate =0.5,n_estimators=300,max_depth=5,gamma=0,subsample=0.8,)
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=35, criterion="entropy")
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=5, criterion="entropy")
lr = LogisticRegression(n_jobs=-1, C=8)  # meta classifier

sclf = StackingCVClassifier(classifiers=[xgb, rfc, etc], meta_classifier=lr, use_probas=True, n_folds=3, verbose=3)

sclf.fit(X, y)
print("training finished")

# save model for later predicting
with open(r'../data/stacking.pkl', 'wb') as f:
    pickle.dump(sclf, f)
print("model dumped")