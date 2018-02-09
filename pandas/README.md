### 此目录用于收集python数据处理的一些用法

```python
# 输出class_id power engine_torque 三个属性列#
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']])
# aliyun_yc_dateset => pandas.DataFrame #
```

> [applymap](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.applymap.html#pandas.DataFrame.applymap) #
> [apply](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html) #
> [transform](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.transform.html#pandas.DataFrame.transform) #

```python
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].applymap(lambda x: '%.2f' % x))

print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(lambda x: x.class_id + x.power + x.engine_torque, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.mean, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.sum, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.max, axis=1))
print(aliyun_yc_dataset[['class_id', 'power', 'engine_torque']].apply(numpy.min, axis=1))

print(aliyun_yc_dataset[['class_id']].transform(lambda x: (x - x.mean()) / x.std()))
```
