"""
示例：多项式贝叶斯
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB

X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])

clf = MultinomialNB()
clf.fit(X, y)
print(clf.predict(X[2:3]))

"""output:
[3]
"""
