
```python
print(ijcai_train_data.isnull().any())   # 查看每个属性列是否有空值 #
print(ijcai_train_data.sample())   # 随机采样 #
print(ijcai_train_data['user_id'].value_counts())   # 查看各个值的个数 #



from imblearn.over_sampling import RandomOverSampler
X = ijcai_train_data.iloc[:, :-1].values  #iloc方法根据位置选择，即选择所有行，所有列去掉右数第一列
y = ijcai_train_data['quality'].values
ros = RandomOverSampler() #构造过采样方法
X, y = ros.fit_sample(X, y)
print(pd.DataFrame(y)[0].value_counts().sort_index()) #得到的x，y是数组，通过DataFrame转化为DataFrame格式

from sklearn.model_selection import cross_val_score
from sklearn import ensemble
#设定随机森林分类模型
rf=ensemble.RandomForestClassifier(100) #设定包含100个决策树
#logistic中的scoring参数指定为accuracy
X = ijcai_train_data.iloc[:, :-1]
y = ijcai_train_data['is_trade']
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(np.mean(scores))

import seaborn as sns
data = ijcai_train_data.corr()
sns.heatmap(data)
plt.show()
```
