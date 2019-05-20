"""
示例：伯努利贝叶斯
"""

import numpy as np
from sklearn.naive_bayes import BernoulliNB

X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])

clf = BernoulliNB()
clf.fit(X, Y)
print(clf.predict(X[2:3]))


"""output:
[3]
"""
