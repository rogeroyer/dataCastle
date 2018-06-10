import math
import numpy as np


def calc(data):
    n = len(data)   # 10000个数
    niu = 0.0   # niu表示平均值,即期望.
    niu2 = 0.0   # niu2表示平方的平均值
    niu3 = 0.0   # niu3表示三次方的平均值
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu /= n
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu*niu)
    return [niu, sigma, niu3]


def calc_stat(data):
    [niu, sigma, niu3] = calc(data)
    n = len(data)
    niu4 = 0.0    # niu4计算峰度计算公式的分子
    for a in data:
        a -= niu
        niu4 += a**4
    niu4 /= n

    skew = (niu3 - 3 * niu*sigma**2-niu**3) / (sigma**3)    # 偏度计算公式
    kurt = niu4/(sigma**4)    # 峰度计算公式:下方为方差的平方即为标准差的四次方
    return [niu, sigma, skew, kurt]
    
    
    
    def calc_skew(data):
    '''计算偏度'''
    if len(data) == 1:
        return 0
    [niu, sigma, niu3] = calc(data)
    skew = (niu3 - 3 * niu * sigma ** 2 - niu ** 3) / (sigma ** 3)  # 偏度计算公式
    return skew


def calc_kurt(data):
    '''计算峰度'''
    if len(data) == 1:
        return 0
    [niu, sigma, niu3] = calc(data)
    n = len(data)
    niu4 = 0.0  # niu4计算峰度计算公式的分子
    for a in data:
        a -= niu
        niu4 += a ** 4
    niu4 /= n
    kurt = niu4 / (sigma ** 4)  # 峰度计算公式:下方为方差的平方即为标准差的四次方
    return kurt
    
    
    
