# !/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import jieba
import jieba.posseg
# import imp

if __name__ == "__main__":
    # imp.reload(sys)
    # sys.setdefaultencoding('utf-8')
    f = open('E:\CODEING\codeingForSelfStudy\ML_Learning_code\\24.HMM\\novel.txt')
    str = f.read()#.decode('utf-8')
    f.close()

    seg = jieba.posseg.cut(str)
    for word, pos in seg:
        # print(word, '|', end=' ')
        print(word, pos, '|', end=' ')
