# !/usr/bin/python
# -*- coding:utf-8 -*-
#24.3.jieba_intro.py
import sys
import jieba
import jieba.posseg
import  io
# from imp import reload


if __name__ == "__main__":
    with io.open('/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/25 jieba/MyBook.txt',
                 'r', encoding='UTF-8') as f:  # 打开新的文本
        line =f.readline()
        while line:
            seg = jieba.posseg.cut(line)
            for word,flag in seg:
                print (word,flag,'|',)
        # line=f.readline()
        f.close()

    # for word, flag in seg:
    #     print (word, flag, '|',)
