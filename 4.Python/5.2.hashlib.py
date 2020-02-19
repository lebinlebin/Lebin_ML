#!/usr/bin/python
"""
信息摘要与安全哈希算法MD5/SHA1
 Message Digest Algorithm/Secure Hash Algorithm

MD5(Message Digest Algorithm)，消息摘要算法，为计算机安全领域广泛使用的一种散
列函数，用以提供消息的完整性保护。
文件号RFC 1321(R.Rivest,MIT Laboratory for Computer Science and RSA Data Security Inc. April 1992)
对于任意长度的信息，经过MD5算法，得到长度为128bit的摘要。

MD5的框架理解
 对于长度为512bit的信息，可以通过处理，得到长度为128bit的摘要。

初始化摘要：
0x0123456789ABCDEFFEDCBA9876543210
 A=0x01234567  B=0x89ABCDEF
 C=0xFEDCBA98  D=0x76543210
 现在的工作，是要用长度为512位的信息，变换初始摘要。


定义变量a,b,c,d,分别记录A,B,C,D;
将512bit的信息按照32bit一组，分成16组；
分别记为Mj (0≤j≤15)；
取某正数s、 tk，定义函数：
FF(a,b,c,d,Mj,s,tk)=(a+F(b,c,d)+Mj+tk)<<s
利用Mj分别进行信息提取，将结果保存到a
 其中， F(X,Y,Z) =(X & Y) | (~X & Z)
"""
import hashlib


if __name__ == "__main__":
    md5 = hashlib.md5()
    md5.update('This is a sentence.'.encode('utf-8'))
    md5.update('This is a second sentence.'.encode('utf-8'))
    print('不出意外，这个将是“乱码”：', md5.digest())#md5.digest() md5是按照十六进制返回的
    print('MD5:', md5.hexdigest())#以十六进制显示

    md5 = hashlib.md5()
    md5.update('This is a sentence.This is a second sentence.'.encode('utf-8'))
    print('MD5:', md5.hexdigest())
    print(md5.digest_size, md5.block_size)
    print('------------------')

    sha1 = hashlib.sha1()
    sha1.update('This is a sentence.'.encode('utf-8'))
    sha1.update('This is a second sentence.'.encode('utf-8'))
    print('不出意外，这个将是“乱码”：', sha1.digest())
    print('SHA1:', sha1.hexdigest())

    sha1 = hashlib.sha1()
    sha1.update('This is a sentence.This is a second sentence.'.encode('utf-8'))
    print('SHA1:', sha1.hexdigest())
    print(sha1.digest_size, sha1.block_size)
    print('=====================')

    md5 = hashlib.new('md5', 'This is a sentence.This is a second sentence.'.encode('utf-8'))
    print(md5.hexdigest())
    sha1 = hashlib.new('sha1', 'This is a sentence.This is a second sentence.'.encode('utf-8'))
    print(sha1.hexdigest())

    print(hashlib.algorithms_available)
