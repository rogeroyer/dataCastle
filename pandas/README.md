### 此目录用于收集python pandas数据处理的一些用法

```python
# 输出class_id power engine_torque 三个属性列#
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']])
# aliyun_yc_dateset => pandas.DataFrame #
# 判断某个值是否属于np.nan #
np.isnan(value)
```

 [pandas.applymap](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.applymap.html#pandas.DataFrame.applymap)
 
 [pandas.apply](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html)
 
 [pandas.transform](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.transform.html#pandas.DataFrame.transform)

```python
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].applymap(lambda x: '%.2f' % x))

print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(lambda x: x.class_id + x.power + x.engine_torque, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.mean, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.sum, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.max, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.min, axis=1))

print(aliyun_yc_dataset[['class_id']].transform(lambda x: (x - x.mean()) / x.std()))
```

```python  
# 通过自定义函数修改变量或者list或者DataFrame的值，原变量的值也会相应改变 #
import pandas as pd

def change(data_set):
    data_set.loc[1, 'id'] = 10

data = pd.DataFrame({'name': [1, 2, 3, 4, 5, 6], 'id': [11, 22, 33, 44, 55, 66]})
print(data)
change(data)
print(data)

Output：
   name  id
0     1  11
1     2  22
2     3  33
3     4  44
4     5  55
5     6  66
   name  id
0     1  11
1     2  22
2     3  33
3     4  44
4     5  10
5     6  66




import pandas as pd
# 但是通过改变切片后的部分DataFrame里面的值，原变量值不变 #
def change(data_set):
    print(data_set)
    data_set.loc[4, 'id'] = 10
    print(data_set)

data = pd.DataFrame({'name': [1, 2, 3, 4, 5, 6], 'id': [11, 22, 33, 44, 55, 66]})
print(data)
change(data[data['name'] > 3])
print(data)

Output：
   name  id
0     1  11
1     2  22
2     3  33
3     4  44
4     5  55
5     6  66
   name  id
3     4  44
4     5  55
5     6  66
   name  id
3     4  44
4     5  10
5     6  66
   name  id
0     1  11
1     2  22
2     3  33
3     4  44
4     5  55
5     6  66
```
