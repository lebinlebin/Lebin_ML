# !/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
x = iris.data[:,:]
y = iris.target
print (x)
print (y)
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x,y)