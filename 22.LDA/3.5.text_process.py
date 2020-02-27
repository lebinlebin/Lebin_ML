#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import jieba
import re
import os
import chardet
# data =open(u"西游记.txt", "rb").read()
# chardet.detect(data)
# {'encoding': 'GB2312', 'confidence': 0.99, 'language': 'Chinese'}


def load_stopwords():
    f = open('/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/22.LDA/stopword.txt','rb+')
    # data = f.read()
    # print(chardet.detect(data))  # 去掉['encoding']可以看完整输出，这里我做了筛选，只显示encoding
    for w in f:
        print(w.strip().decode('GB18030'))
        stopwords.add(w.strip().decode('GB18030'))
    f.close()


def segment_one_file(input_file_name, output_file_name):
    f = open(input_file_name,'rb+')#mode='r'
    f_output = open(output_file_name, mode='w')
    pattern = re.compile('<content>(.*?)</content>')
    for line in f:
        line = line.strip().decode('GB18030')
        # print(line)
        news = re.findall(pattern=pattern, string=line)
        for one_news in news:
            words_list = []
            words = jieba.cut(one_news.strip())
            for word in words:
                word = word.strip()
                if word not in stopwords:
                    words_list.append(word)
            if len(words_list) > 10:
                s = u' '.join(words_list)
                f_output.write(s + '\n') #decode()
    f.close()
    f_output.close()

if __name__ == "__main__":
    stopwords = set()
    load_stopwords()
    print("load  stopwords  sucess!!! ")

    input_dir = '/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/22.LDA/200806/SogouCA.reduced'
    output_dir = '/Users/liulebin/Documents/codeing/codeingForSelfStudy/ML-Basic-Theory-Study/ML_Learning_code/22.LDA/200806_segment'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file_name in os.listdir(input_dir):
        if os.path.splitext(file_name)[-1] == '.txt':
            print(file_name)
            segment_one_file(os.path.join(input_dir, file_name), os.path.join(output_dir, file_name))
