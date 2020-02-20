# coding:utf-8

import re
import requests
import os


if __name__ == '__main__':
    word = u'中关村'
    picture_path = word
    if not os.path.exists(picture_path):
        os.mkdir(picture_path)

    url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&ct=201326592&v=flip'
    result = requests.get(url)
    pic_url = re.findall('"objURL":"(.*?)",', result.text, re.S)  #re: regular expression
    print ('开始下载...')
    for i, url in enumerate(pic_url, start=1):
        print ('正在下载第'+str(i)+'张图片：', url)
        try:
            pic = requests.get(url, timeout=100)
        except requests.exceptions.ConnectionError as requests.exceptions.ReadTimeout:
            print ('当前图片无法下载', url)
            continue
        fp = open(picture_path + '\\'+ str(i) + '.jpg', 'wb')
        fp.write(pic.content)
        fp.close()
