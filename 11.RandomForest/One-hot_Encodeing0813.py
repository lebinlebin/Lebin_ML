# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


if __name__ == '__main__':
    pd.set_option('display.width', 500)
    col_names = 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    data = pd.read_csv('adult.data', header=None, names=col_names)

    # 丢弃无效数据
    # data.drop('fnlwgt', axis=1, inplace=True)

    # 离散化
    data['age'] = pd.cut(data['age'], bins=10, labels=np.arange(10))
    bins = -1, 1, 100, 300, 1000, 5000, 10000, 30000, 50000
    data.loc[data['capital-gain'] > 80000, 'capital-gain'] = 0
    data['capital-gain'] = pd.cut(data['capital-gain'], bins=bins, labels=np.arange(len(bins)-1))
    bins = -1, 1, 100, 300, 1000, 5000
    data['capital-loss'] = pd.cut(data['capital-loss'], bins=bins, labels=np.arange(len(bins)-1))
    data['hours-per-week'] = pd.cut(data['hours-per-week'], bins=10, labels=np.arange(10))

    # one-hot
    le = LabelEncoder()
    lb = LabelBinarizer()
    for col in data.columns[:-1]:
            data[col] = le.fit_transform(data[col])
            a = lb.fit_transfrm(data[col])
            col_names_onehot = [col + '_' + str(i) for i in range(a.shape[1])]
            a = pd.DataFrame(data=a, columns=col_names_onehot)
            data = pd.concat((data, a), axis=1)
            data.drop(col, axis=1, inplace=True)

   # Y
    print (data)
    data['income'] = LabelEncoder().fit_transform(data['income'])

    x = data[data.columns[:-1]]
    y = data['income']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=16, min_samples_split=5, min_samples_leaf=3)
    # model = DecisionTreeClassifier(criterion='gini', min_samples_leaf=5)
    model.fit(x_train, y_train)
    print (col_names)
    print (model.feature_importances_)
    y_train_pred = model.predict(x_train)
    print ('RF训练集准确率/错误率：', accuracy_score(y_train, y_train_pred))
    _test_pred = model.predict(x_test)
    print ('RF测试集准确率/错误率：', accuracy_score(y_test, _test_pred))

    nb = MultinomialNB(alpha=1.0)
    nb.fit(x_train, y_train)
    y_train_pred = nb.predict(x_train)
    print ('NB训练集准确率/错误率：', accuracy_score(y_train, y_train_pred))
    y_test_pred = nb.predict(x_test)
    print ('NB测试集准确率/错误率：', accuracy_score(y_test, y_test_pred))

    lr = LogisticRegressionCV(penalty='l2', Cs=np.logspace(-1,2,10))
    lr.fit(x_train, y_train)
    print ('最优参数：', lr.C_)
    print ('参数：', lr.coef_, lr.intercept_)
    y_train_pred = lr.predict(x_train)
    print ('LR训练集准确率/错误率：', 1 - accuracy_score(y_train, y_train_pred))
    y_test_pred = lr.predict(x_test)
    print ('LR测试集准确率/错误率：', 1 - accuracy_score(y_test, y_test_pred))
