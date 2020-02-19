"""
想要原本密集的区间的值尽可能的分散，原本分散的区间的值尽量的聚合。
在图像处理领域，有一类算法天然是为了这类目的存在的：对比度增强算法
常用的对比度增强算法有对数变换、指数变换（伽马变换）、灰度拉升等，
至于直方图均衡化甚至是小波变换等一些更加复杂的算法，由于它们均依赖于数据的先验分布，因此我们暂时不作考虑：
"""
import  math
def domain_transform(ctr, func):
    a = 0.002
    b = 0.3
    ctr = max(a, ctr)
    ctr = min(b, ctr)
    ctr = (ctr - a) / (b - a)
    fa, fb = func(0), func(1)
    assert fa >= 0
    return (func(ctr) - fa) / (fb - fa)
def map_func1(ctrs):
    return [domain_transform(ctr, lambda x: x) for ctr in ctrs]
def map_func2(ctrs):
    return [domain_transform(ctr, lambda x: math.log(1 + 100 * x)) for ctr in ctrs]
def map_func3(ctrs):
    return [domain_transform(ctr, lambda x: math.sqrt(x)) for ctr in ctrs]
def map_func4(ctrs):
    return [mat2gray(ctr) for ctr in ctrs]
def mat2gray(x):
    return x / (x + 0.01)