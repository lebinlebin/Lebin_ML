"""
三分法示意图：先取 [L,R] 的中点 mid，再取 [mid,R] 的中点 mmid，通过比较 f(mid) 与 f(mmid) 的大小来缩小范围，当最后 L=R-1 时，再比较下这两个点的值，我们就找到了答案。
"""

import numpy
import math

def loss_func(theta, wtx, labels):
    size = len(wtx)
    assert size == len(labels)
    y = 1 / (1 + numpy.exp(- wtx + theta))
    log_loss = - labels * numpy.log(y) - (1 - labels) * numpy.log(1 - y)
    return numpy.sum(log_loss) / size

def solve_theta(wtx, labels):
    left, right = -5.0, 5.0
    delta = 1.0
    while delta >= 0.001:
        mid = (left + right) / 2
        mmid = (mid + right) / 2
        f1 = loss_func(mid, wtx, labels)
        f2 = loss_func(mmid, wtx, labels)
        if f1 < f2:
            right = mmid
        else:
            left = mid
        delta = right - left
    return left