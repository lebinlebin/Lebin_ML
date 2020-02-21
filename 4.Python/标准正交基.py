#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
标准正交基
"""
from scipy.linalg import orth
from numpy.linalg import  matrix_rank
import numpy as np
a = (np.eye(1), np.diag((1.,2.,3.)), np.arange(9,dtype=np.float).reshape((3,3)))
for m in a:
    print(m,u'的秩为：',matrix_rank(m))
    print(u'正交基为：\n',orth(m))