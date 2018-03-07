### 注意事项
- 学正:

    - 问题1）train 数据集中有特征字段 和 真实的购买标记 (is_trade)；test 数据集只给出了特征字段，没有给出 is_trade，需要预测发生购买的概率。

    - 问题2）初赛和复赛期间有几个时间节点会切换数据。每一批数据的分布情况可能不同，需要根据数据的分布情况，判断是哪一类挑战，然后优化算法模型。

    - 问题3）比如 1970/01/01 00:00:00 这个时间，通过偏移处理，转换成了 1970/01/23 00:00:00. 与原始时间戳相比，偏移后的时间戳在时分秒上不变，只在日期上发生变化。

> 用于初赛的数据包含了若干天的样本。最后一天的数据用于结果评测，对选手不公布；其余日期的数据作为训练数据，提供给参赛选手。

> 训练集时间：20180918 - 20180924   测试集时间:20180925 - 20180925

```
instance_id
code:
print(ijcai_train_data.shape)
print(ijcai_train_data[['instance_id']].drop_duplicates(['instance_id']).shape)
result:
(478138, 27)
(478087, 1)

item_id
code:
print(ijcai_train_data.shape)
print(ijcai_train_data[['item_id']].drop_duplicates(['item_id']).shape)
result:
(478138, 27)
(10075, 1)

shop_id
code:
print(ijcai_train_data.shape)
print(ijcai_train_data[['shop_id']].drop_duplicates(['shop_id']).shape)
result:
(478138, 27)
(3959, 1)

user_id
code:
print(ijcai_train_data.shape)
print(ijcai_train_data[['user_id']].drop_duplicates(['user_id']).shape)
result:
(478138, 27)
(197694, 1)
```
