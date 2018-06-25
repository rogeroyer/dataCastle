### **[参数设置](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)**

<table>
<tr>
	<td>Parameters:</td>
<td style="overflow:scroll;">
<pre>
<strong>n_clusters</strong> : int, optional, default: 8

The number of clusters to form as well as the number of centroids to generate.

init : {‘k-means++’, ‘random’ or an ndarray}

Method for initialization, defaults to ‘k-means++’:

‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.

‘random’: choose k observations (rows) at random from data for the initial centroids.

If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

<strong>n_init </strong>: int, default: 10

Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

<strong>max_iter </strong>: int, default: 300

Maximum number of iterations of the k-means algorithm for a single run.

tol : float, default: 1e-4

Relative tolerance with regards to inertia to declare convergence

precompute_distances : {‘auto’, True, False}

Precompute distances (faster but takes more memory).

‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead per job using double precision.

True : always precompute distances

False : never precompute distances

verbose : int, default 0

Verbosity mode.

<strong>random_state</strong> : int, RandomState instance or None, optional, default: None

If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

<strong>copy_x</strong> : boolean, default True

When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True, then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean.

<strong>n_jobs </strong>: int

The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel.

If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.

<strong>algorithm</strong> : “auto”, “full” or “elkan”, default=”auto”

K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. “auto” chooses “elkan” for dense data and “full” for sparse data.
</pre>
</td>
</tr>
<tr>
<td>Attributes:</td>
<td style="overflow:scroll;">
<pre>
<strong>cluster_centers_ </strong>: array, [n_clusters, n_features]

Coordinates of cluster centers

<strong>labels_</strong> : :

Labels of each point

<strong>inertia_ </strong>: float

Sum of squared distances of samples to their closest cluster center.
</pre>
</td>
</tr>
</table>
***

### **参数详解[中文]**

```
sklearn.cluster.KMeans(
    n_clusters=8,
    init='k-means++', 
    n_init=10, 
    max_iter=300, 
    tol=0.0001, 
    precompute_distances='auto', 
    verbose=0, 
    random_state=None, 
    copy_x=True, 
    n_jobs=1, 
    algorithm='auto'
    )
    
n_clusters: 簇的个数，即你想聚成几类
init: 初始簇中心的获取方法
n_init: 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初始10个质心，实现算法，然后返回最好的结果。
max_iter: 最大迭代次数（因为kmeans算法的实现需要迭代）
tol: 容忍度，即kmeans运行准则收敛的条件
precompute_distances:是否需要提前计算距离，这个参数会在空间和时间之间做权衡，如果是True 会把整个距离矩阵都放到内存中，auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的
verbose: 冗长模式（不太懂是啥意思，反正一般不去改默认值）
random_state: 随机生成簇中心的状态条件。
copy_x: 对是否修改数据的一个标记，如果True，即复制了就不会修改数据。bool 在scikit-learn 很多接口中都会有这个参数的，就是是否对输入数据继续copy 操作，以便不修改用户的输入数据。这个要理解Python 的内存机制才会比较清楚。
n_jobs: 并行设置
algorithm: kmeans的实现算法，有：’auto’, ‘full’, ‘elkan’, 其中 ‘full’表示用EM方式实现
虽然有很多参数，但是都已经给出了默认值。所以我们一般不需要去传入这些参数,参数的。可以根据实际需要来调用。
```

***
### **程序示例**

- Numpy
```
>>> from sklearn.cluster import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 4], [4, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([0, 0, 0, 1, 1, 1], dtype=int32)
>>> kmeans.predict([[0, 0], [4, 4]])
array([0, 1], dtype=int32)
>>> kmeans.cluster_centers_
array([[ 1.,  2.],
       [ 4.,  2.]])
```

- Pandas

```
#coding=utf-8

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
test_data = pd.read_csv('test.csv', low_memory=False, header=None)

# X = test_data.as_matrix()  # 将pandas.DataFrame转化为numpy.array() #
# X = numpy.array(test_data)   # 将pandas.DataFrame转化为numpy.array() #
X = test_data.values   # 将pandas.DataFrame转化为numpy.array() #

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print('类簇标签', kmeans.labels_)   # 计算类簇标签 #

print(kmeans.predict([[489877, 42, 1], [490515, 3117, 3]]))   # 预测类簇的标签 list #

print('类簇中心', kmeans.cluster_centers_)
```
***
### **聚类实例**
#### 代码
```
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('test_data.csv', low_memory=False, nrows=10000).values
estimator = KMeans(n_clusters=5)
res = estimator.fit(data)
lable_pred = estimator.labels_
centroids = estimator.cluster_centers_
inertia = estimator.inertia_

print(lable_pred)
print(centroids)
print(inertia)

for i in range(len(data)):
    if int(lable_pred[i]) == 0:
        plt.scatter(data[i][0], data[i][1], color='red')
    if int(lable_pred[i]) == 1:
        plt.scatter(data[i][0], data[i][1], color='black')
    if int(lable_pred[i]) == 2:
        plt.scatter(data[i][0], data[i][1], color='blue')
    if int(lable_pred[i]) == 3:
        plt.scatter(data[i][0], data[i][1], color='orange')
    if int(lable_pred[i]) == 4:
        plt.scatter(data[i][0], data[i][1], color='yellow')
plt.show()
```
#### 运行结果
![这里写图片描述](http://img.blog.csdn.net/20180305153637212?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcm9nZXJfcm95ZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

***
### **参考资料**

[`伯乐在线`](http://python.jobbole.com/88535/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	[`sklearn`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

