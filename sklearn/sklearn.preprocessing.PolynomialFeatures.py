"""
使用sklearn.preprocessing.PolynomialFeatures来进行特征的构造。

它是使用多项式的方法来进行的，如果有a，b两个特征，那么它的2次多项式为（1,a,b,a^2,ab, b^2）。

PolynomialFeatures有三个参数

degree：控制多项式的度

interaction_only： 默认为False，如果指定为True，那么就不会有特征自己和自己结合的项，上面的二次项中没有a^2和b^2。

include_bias：默认为True。如果为True的话，那么就会有上面的 1那一项。
"""

>>> import numpy as np
>>> np.arange(6)
array([0, 1, 2, 3, 4, 5])
>>> X = np.arange(6).reshape(3, 2)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5]])

>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly = PolynomialFeatures(2)
>>> poly.fit_transform(X)
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
       
>>> poly = PolynomialFeatures(3)
>>> poly.fit_transform(X)
array([[  1.,   0.,   1.,   0.,   0.,   1.,   0.,   0.,   0.,   1.],
       [  1.,   2.,   3.,   4.,   6.,   9.,   8.,  12.,  18.,  27.],
       [  1.,   4.,   5.,  16.,  20.,  25.,  64.,  80., 100., 125.]])
