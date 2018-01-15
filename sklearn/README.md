## sklearn python API

- [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
```python
from sklearn.linear_model import LinearRegression         # 线性回归 #
module = LinearRegression()
module.fit(x, y)
module.score(x, y)
module.predict(test)
```
***

- [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
```python
from sklearn.linear_model import LogisticRegression         # 逻辑回归 #
module = LogisticRegression()
module.fit(x, y)
module.score(x, y)
module.predict(test)
```
***

- [KNN](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
```python
from sklearn.neighbors import KNeighborsClassifier     #K近邻#
from sklearn.neighbors import KNeighborsRegressor
module = KNeighborsClassifier(n_neighbors=6)
module.fit(x, y)
predicted = module.predict(test)
predicted = module.predict_proba(test)
```
***

- [SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
```python
from sklearn import svm                                #支持向量机#
module = svm.SVC()
module.fit(x, y)
module.score(x, y)
module.predict(test)
module.predict_proba(test)
```
***


- [naive_bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
```python
from sklearn.naive_bayes import GaussianNB            #朴素贝叶斯分类器#
module = GaussianNB()
module.fit(x, y)
predicted = module.predict(test)
```
***

- [DecisionTree](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
```python
from sklearn import tree                              #决策树分类器#
module = tree.DecisionTreeClassifier(criterion='gini')
module.fit(x, y)
module.score(x, y)
module.predict(test)
```
***

- [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
```python
from sklearn.cluster import KMeans                    #kmeans聚类#
module = KMeans(n_clusters=3, random_state=0)
module.fit(x, y)
module.predict(test)
```
***

- [RandomForest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
```python
from sklearn.ensemble import RandomForestClassifier  #随机森林#
from sklearn.ensemble import RandomForestRegressor
module = RandomForestClassifier()
module.fit(x, y)
module.predict(test)
```
***

- [GBDT](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
```python
from sklearn.ensemble import GradientBoostingClassifier      #Gradient Boosting 和 AdaBoost算法#
from sklearn.ensemble import GradientBoostingRegressor
module = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
module.fit(x, y)
module.predict(test)
```
***

- [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
```python
from sklearn.decomposition import PCA              #PCA特征降维#
train_reduced = PCA.fit_transform(train)
test_reduced = PCA.transform(test)
```
***

### [References](https://mp.weixin.qq.com/s/vwNaLDkmXiNRmj-D3Ae6eQ)
