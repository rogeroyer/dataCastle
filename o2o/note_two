#coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

# read_data = pd.read_csv(r'D:\aliyun\ccf_offline_stage1_train.csv', header=None)
# read_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

# 划分数据集 #
# read_data = read_data[read_data['date_received'] != 'null']
# # train_data = read_data[((read_data['date_received'] >= '20160401') & (read_data['date_received'] < '20160615')) | (read_data['date_received'] == 'null')]
# train_data = read_data[(read_data['date_received'] >= '20160401') & (read_data['date_received'] < '20160615')]
# print train_data
# label_data = read_data[(read_data['date_received'] >= '20160515') & (read_data['date_received'] < '20160615')]
# # 给未消费数据打标为0 #
# label_data_part1 = label_data[label_data['date'] == 'null']
# label_data_part1['label'] = 0
# # 给领券消费的数据打标 #
# label_data_part2 = label_data[label_data['date'] != 'null']
# label_data_part2['date_received'] = pd.to_datetime(label_data_part2['date_received'])
# label_data_part2['date'] = pd.to_datetime(label_data_part2['date'])
# # 领了优惠券并在15天之内消费的打标为1否则为0 #
# label_data_part2['label'] = [0 if int(i.days) > 15 else 1 for i in (label_data_part2['date'] - label_data_part2['date_received'])]
# #去连接线- 将日期格式转换为文本格式#
# label_data_part2['date_received'] = [str(i)[:10].replace('-', '') for i in label_data_part2['date_received']]
# label_data_part2['date'] = [str(i)[:10].replace('-', '') for i in label_data_part2['date']]
# # 合并已经贴好标签的两个数据集 #
# label_data = label_data_part1.append(label_data_part2)
# train_data.to_csv('train_data.csv', index=None)
# label_data.to_csv('label_data.csv', index=None)

####################################################################
####################################################################
####################################################################
train_data = pd.read_csv(r'train_data.csv', header=None)
train_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
train_data['date_received'] = [str(index) for index in train_data['date_received']]

###############################################################################
# 每个商户发了多少优惠券 #
feature1 = pd.pivot_table(train_data, index='merchant_id', values='coupon_id', aggfunc='count')
feature1['merchant_id'] = feature1.index
feature1.columns = ['feature1', 'merchant_id']

# # 每位用户领了多少张优惠券 #
feature2 = pd.pivot_table(train_data[train_data['coupon_id'].notnull()], index='user_id', values='coupon_id', aggfunc='count')
feature2['user_id'] = feature2.index
feature2.columns = ['feature2', 'user_id']

# print feature2
# feature2.plot(kind='hist')
# plt.show()
# feature = pd.pivot_table(train_data[train_data['coupon_id'].notnull()], index='user_id', values='date', aggfunc='count')
# feature2['feature3'] = feature['date']
# print feature2[feature2['feature3'] != feature2['feature2']]

# 商户距离 3个 # # 均值，中位数，最大，最小 再两两交叉加减乘，本身与本身加减乘#
distance_means = train_data[train_data['distance'] != 'null']
distance_means['distance'] = [int(i) for i in distance_means['distance']]
distance = distance_means['distance'].median() #中位数#
train_data['distance'] = train_data['distance'].replace('null', distance)
train_data['distance'] = [int(i) for i in train_data['distance']]
feature3 = pd.pivot_table(train_data, index='user_id', values='distance', aggfunc='mean')
feature3['user_id'] = feature3.index
feature3.columns = ['feature3', 'user_id']
feature = pd.pivot_table(train_data, index='user_id', values='distance', aggfunc='max')
feature3['one'] = feature['distance']
feature = pd.pivot_table(train_data, index='user_id', values='distance', aggfunc='min')
feature3['two'] = feature['distance']
feature = pd.pivot_table(train_data, index='user_id', values='distance', aggfunc='median')
feature3['three'] = feature['distance']
feature3['four'] = feature3['feature3'] + feature3['one']
feature3['four1'] = feature3['feature3'] - feature3['one']
feature3['four2'] = feature3['feature3'] * feature3['one']
feature3['five'] = feature3['feature3'] + feature3['two']
feature3['five1'] = feature3['feature3'] - feature3['two']
feature3['five2'] = feature3['feature3'] * feature3['two']
feature3['six'] = feature3['feature3'] + feature3['three']
feature3['six1'] = feature3['feature3'] - feature3['three']
feature3['six2'] = feature3['feature3'] * feature3['three']
feature3['seven'] = feature3['one'] + feature3['two']
feature3['seven1'] = feature3['one'] - feature3['two']
feature3['seven2'] = feature3['one'] * feature3['two']
feature3['eight'] = feature3['one'] * feature3['three']
feature3['eight1'] = feature3['one'] * feature3['three']
feature3['eight2'] = feature3['one'] * feature3['three']
feature3['nine'] = feature3['two'] * feature3['three']
feature3['nine1'] = feature3['two'] * feature3['three']
feature3['nine2'] = feature3['two'] * feature3['three']
feature3['ten'] = feature3['one'] * feature3['one']
feature3['ten1'] = feature3['one'] * feature3['one']
feature3['ten1'] = feature3['one'] * feature3['one']
feature3['eleven'] = feature3['feature3'] * feature3['feature3']
feature3['eleven1'] = feature3['feature3'] * feature3['feature3']
feature3['eleven2'] = feature3['feature3'] * feature3['feature3']
feature3['twelve'] = feature3['two'] * feature3['two']
feature3['twelve1'] = feature3['two'] * feature3['two']
feature3['twelve2'] = feature3['two'] * feature3['two']
feature3['thirteen'] = feature3['three'] * feature3['three']
feature3['thirteen1'] = feature3['three'] * feature3['three']
feature3['thirteen2'] = feature3['three'] * feature3['three']
# print feature3

# 训练集提取日期相关特征 #
# train_data['date_received'] = pd.to_datetime(train_data['date_received'])
# train_data['week'] = [index.weekday()+1 for index in train_data['date_received']]
# feature = train_data.iloc[:, [0, 7]]
# feature['week1'] = feature['week']
# feature12 = pd.pivot_table(feature, index=['user_id', 'week'], values='week1', aggfunc='sum') #mode   , values='week', aggfunc='mode'
# print feature12

# print train_data['week'].mode()

# 组合特征，从已经提取的特征里面交叉进行加减乘除 #
# 最以后一次领券，第一次领券 #


#########################################################################
#user_id单个属性提取特征 #
train_data['user_id1'] = train_data['user_id']
feature8 = pd.pivot_table(train_data, index='user_id', values='user_id1', aggfunc='count')
feature8['user_id'] = feature8.index
train_data = train_data.drop('user_id1', axis=1)

train_data['merchant_id1'] = train_data['merchant_id']
feature9 = pd.pivot_table(train_data, index='merchant_id', values='merchant_id1', aggfunc='count')
feature9['merchant_id'] = feature9.index
train_data = train_data.drop('merchant_id1', axis=1)

train_data['coupon_id1'] = train_data['coupon_id']
feature10 = pd.pivot_table(train_data, index='coupon_id', values='coupon_id1', aggfunc='count')
feature10['coupon_id'] = feature10.index
train_data = train_data.drop('coupon_id1', axis=1)

#####################################################################
#####################################################################
# # 从打标的数据集中提取特征 #
# # 读取已经打标的数据集 #
label_data = pd.read_csv('label_data.csv', header=None)
label_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date', 'label']
# print label_data

# 优惠率 #
feature4 = label_data.iloc[:, [2, 3]]
feature4['discount_rate'] = [index if index[0] == '0' else (int(str(index).split(':')[1])*1.0)/int(str(index).split(':')[0])
                                           for index in feature4['discount_rate']]
feature4.columns = ['coupon_id', 'feature4']
# print feature4

# 满多少 #
feature5 = label_data.iloc[:, [2, 3]]
feature5.columns = ['coupon_id', 'up']
feature5['up'] = [index if index[0] == '0' else int(str(index).split(':')[0]) for index in feature5['up']]
up_median = feature5['up'].median()
feature5['up'] = [up_median if str(index)[0] == '0' else index for index in feature5['up']]
# print feature5

# 减多少 #
feature6 = label_data.iloc[:, [2, 3]]
feature6.columns = ['coupon_id', 'down']
feature6['down'] = [index if index[0] == '0' else int(str(index).split(':')[1]) for index in feature6['down']]
up_median = feature6['down'].median()
feature6['down'] = [up_median if str(index)[0] == '0' else index for index in feature6['down']]
# print feature6

# 折扣类型 #
feature7 = label_data.iloc[:, [2, 3]]
feature7.columns = ['coupon_id', 'feature7']
feature7['feature7'] = [0 if index[0] == '0' else 1 for index in feature7['feature7']]

# 领券日期相关特征 9个 #
label_data['date_received'] = pd.to_datetime(train_data['date_received'])
label_data['week'] = [index.weekday()+1 for index in label_data['date_received']]
feature11 = label_data.iloc[:, [8]]
feature11['one'] = [1 if index == 1 else 0 for index in feature11['week']]
feature11['two'] = [1 if index == 2 else 0 for index in feature11['week']]
feature11['three'] = [1 if index == 3 else 0 for index in feature11['week']]
feature11['four'] = [1 if index == 4 else 0 for index in feature11['week']]
feature11['five'] = [1 if index == 5 else 0 for index in feature11['week']]
feature11['six'] = [1 if index == 6 else 0 for index in feature11['week']]
feature11['seven'] = [1 if index == 7 else 0 for index in feature11['week']]
set_one = set([7, 6])
feature11['eight'] = [1 if index in set_one else 0 for index in feature11['week']]
# print feature11

# 判断是否只领了一次券 #
feature = label_data.iloc[:, [0, 5]]
feature13 = pd.pivot_table(feature, index='user_id', values='date_received', aggfunc='count')
feature13['user_id'] = feature13.index
feature13['feature13'] = [1 if index == 1 else 0 for index in feature13['date_received']]
feature13 = feature13.drop('date_received', axis=1)
# print feature13

#########################################################################
label_data = label_data.iloc[:, [0, 1, 2, 7]]
label_data = label_data.merge(feature1, on='merchant_id', how='left')
label_data = label_data.merge(feature2, on='user_id', how='left')
label_data = label_data.merge(feature3, on='user_id', how='left')
label_data['feature4'] = feature4['feature4']
label_data['feature5'] = feature5['up']
label_data['feature6'] = feature6['down']
label_data['feature7'] = feature7['feature7']
label_data = label_data.merge(feature8, on='user_id', how='left')
label_data = label_data.merge(feature9, on='merchant_id', how='left')
label_data = label_data.merge(feature10, on='coupon_id', how='left')
label_data['feature11_week'] = feature11['week']
label_data['feature11_one'] = feature11['one']
label_data['feature11_two'] = feature11['two']
label_data['feature11_three'] = feature11['three']
label_data['feature11_four'] = feature11['four']
label_data['feature11_five'] = feature11['five']
label_data['feature11_six'] = feature11['six']
label_data['feature11_seven'] = feature11['seven']
label_data['feature11_eight'] = feature11['eight']
label_data = label_data.merge(feature13, on='user_id', how='left')
# print label_data


# label_data.iloc[:, 3].to_csv('label.csv', index=None, header=None)
# label_data.iloc[:, 4:-1].to_csv('train.csv', index=None, header=None)



# 判断日期是否是周末 #
# train_data = train_data[train_data['date'] != 'null']
# train_data['date_received'] = pd.to_datetime(train_data['date_received'])
# train_data['week'] = [index.weekday()+1 for index in train_data['date_received']]
# train_data['date'] = pd.to_datetime(train_data['date'])
# train_data['week'] = [index.weekday()+1 for index in train_data['date']]
# week = list(train_data[train_data['date'] != 'null']['week'])
# plt.hist(week, color='yellow')
# plt.show()
# print week



###############################################################
###############################################################
###############################################################
test_dataset = pd.read_csv(r'D:\aliyun\ccf_offline_stage1_test_revised.csv', header=None)
test_dataset.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
test_dataset_two = test_dataset.copy()
# 每个商户发了多少优惠券 #
feature1 = pd.pivot_table(test_dataset, index='merchant_id', values='coupon_id', aggfunc='count')
feature1['merchant_id'] = feature1.index
feature1.columns = ['feature1', 'merchant_id']

# # 每位用户领了多少张优惠券 #
feature2 = pd.pivot_table(test_dataset[test_dataset['coupon_id'].notnull()], index='user_id', values='coupon_id', aggfunc='count')
feature2['user_id'] = feature2.index
feature2.columns = ['feature2', 'user_id']

# 商户距离 3个 #
distance_means = test_dataset[test_dataset['distance'] != 'null']
distance_means['distance'] = [int(i) for i in distance_means['distance']]
distance = distance_means['distance'].median() #中位数#
test_dataset['distance'] = test_dataset['distance'].replace('null', distance)
test_dataset['distance'] = [int(i) for i in test_dataset['distance']]

feature3 = pd.pivot_table(test_dataset, index='user_id', values='distance', aggfunc='mean')
feature3['user_id'] = feature3.index
feature3.columns = ['feature3', 'user_id']
feature = pd.pivot_table(test_dataset, index='user_id', values='distance', aggfunc='max')
feature3['one'] = feature['distance']
feature = pd.pivot_table(test_dataset, index='user_id', values='distance', aggfunc='min')
feature3['two'] = feature['distance']
feature = pd.pivot_table(test_dataset, index='user_id', values='distance', aggfunc='median')
feature3['three'] = feature['distance']
feature3['four'] = feature3['feature3'] + feature3['one']
feature3['four1'] = feature3['feature3'] - feature3['one']
feature3['four2'] = feature3['feature3'] * feature3['one']
feature3['five'] = feature3['feature3'] + feature3['two']
feature3['five1'] = feature3['feature3'] - feature3['two']
feature3['five2'] = feature3['feature3'] * feature3['two']
feature3['six'] = feature3['feature3'] + feature3['three']
feature3['six1'] = feature3['feature3'] - feature3['three']
feature3['six2'] = feature3['feature3'] * feature3['three']
feature3['seven'] = feature3['one'] + feature3['two']
feature3['seven1'] = feature3['one'] - feature3['two']
feature3['seven2'] = feature3['one'] * feature3['two']
feature3['eight'] = feature3['one'] * feature3['three']
feature3['eight1'] = feature3['one'] * feature3['three']
feature3['eight2'] = feature3['one'] * feature3['three']
feature3['nine'] = feature3['two'] * feature3['three']
feature3['nine1'] = feature3['two'] * feature3['three']
feature3['nine2'] = feature3['two'] * feature3['three']
feature3['ten'] = feature3['one'] * feature3['one']
feature3['ten1'] = feature3['one'] * feature3['one']
feature3['ten1'] = feature3['one'] * feature3['one']
feature3['eleven'] = feature3['feature3'] * feature3['feature3']
feature3['eleven1'] = feature3['feature3'] * feature3['feature3']
feature3['eleven2'] = feature3['feature3'] * feature3['feature3']
feature3['twelve'] = feature3['two'] * feature3['two']
feature3['twelve1'] = feature3['two'] * feature3['two']
feature3['twelve2'] = feature3['two'] * feature3['two']
feature3['thirteen'] = feature3['three'] * feature3['three']
feature3['thirteen1'] = feature3['three'] * feature3['three']
feature3['thirteen2'] = feature3['three'] * feature3['three']
# print feature3

# ####################################################
#user_id单个属性提取特征 #
test_dataset['user_id1'] = test_dataset['user_id']
feature8 = pd.pivot_table(test_dataset, index='user_id', values='user_id1', aggfunc='count')
feature8['user_id'] = feature8.index
test_dataset = test_dataset.drop('user_id1', axis=1)
# print feature8
test_dataset['merchant_id1'] = test_dataset['merchant_id']
feature9 = pd.pivot_table(test_dataset, index='merchant_id', values='merchant_id1', aggfunc='count')
feature9['merchant_id'] = feature9.index
test_dataset = test_dataset.drop('merchant_id1', axis=1)
# print feature9
test_dataset['coupon_id1'] = test_dataset['coupon_id']
feature10 = pd.pivot_table(test_dataset, index='coupon_id', values='coupon_id1', aggfunc='count')
feature10['coupon_id'] = feature10.index
test_dataset = test_dataset.drop('coupon_id1', axis=1)
# print feature10
######################################################
# 优惠率 #
feature4 = test_dataset.iloc[:, [2, 3]]
feature4['discount_rate'] = [index if index[0] == '0' else (int(str(index).split(':')[1])*1.0)/int(str(index).split(':')[0])
                                           for index in feature4['discount_rate']]
feature4.columns = ['coupon_id', 'feature4']
# print feature4

# 满多少 #
feature5 = test_dataset.iloc[:, [2, 3]]
feature5.columns = ['coupon_id', 'up']
feature5['up'] = [index if index[0] == '0' else int(str(index).split(':')[0]) for index in feature5['up']]
up_median = feature5['up'].median()
feature5['up'] = [up_median if str(index)[0] == '0' else index for index in feature5['up']]
# print feature5

# 减多少 #
feature6 = test_dataset.iloc[:, [2, 3]]
feature6.columns = ['coupon_id', 'down']
feature6['down'] = [index if index[0] == '0' else int(str(index).split(':')[1]) for index in feature6['down']]
up_median = feature6['down'].median()
feature6['down'] = [up_median if str(index)[0] == '0' else index for index in feature6['down']]
# print feature6

# 折扣类型 #
feature7 = test_dataset.iloc[:, [2, 3]]
feature7.columns = ['coupon_id', 'feature7']
feature7['feature7'] = [0 if index[0] == '0' else 1 for index in feature7['feature7']]
# print feature7

# 领券日期相关特征 9个 #
test_dataset['date_received'] = [str(index) for index in test_dataset['date_received']]
test_dataset['date_received'] = pd.to_datetime(test_dataset['date_received'])
test_dataset['week'] = [index.weekday()+1 for index in test_dataset['date_received']]
feature11 = test_dataset.iloc[:, [6]]
feature11['one'] = [1 if index == 1 else 0 for index in feature11['week']]
feature11['two'] = [1 if index == 2 else 0 for index in feature11['week']]
feature11['three'] = [1 if index == 3 else 0 for index in feature11['week']]
feature11['four'] = [1 if index == 4 else 0 for index in feature11['week']]
feature11['five'] = [1 if index == 5 else 0 for index in feature11['week']]
feature11['six'] = [1 if index == 6 else 0 for index in feature11['week']]
feature11['seven'] = [1 if index == 7 else 0 for index in feature11['week']]
set_one = set([7, 6])
feature11['eight'] = [1 if index in set_one else 0 for index in feature11['week']]
# print feature11

# 判断是否只领了一次券 #
feature = test_dataset.iloc[:, [0, 5]]
feature13 = pd.pivot_table(feature, index='user_id', values='date_received', aggfunc='count')
feature13['user_id'] = feature13.index
feature13['feature13'] = [1 if index == 1 else 0 for index in feature13['date_received']]
feature13 = feature13.drop('date_received', axis=1)
# print feature13

################################################

test_dataset = test_dataset.iloc[:, [0, 1, 2]]
test_dataset = test_dataset.merge(feature1, on='merchant_id', how='left')
test_dataset = test_dataset.merge(feature2, on='user_id', how='left')
test_dataset = test_dataset.merge(feature3, on='user_id', how='left')
test_dataset['feature4'] = feature4['feature4']
test_dataset['feature5'] = feature5['up']
test_dataset['feature6'] = feature6['down']
test_dataset['feature7'] = feature7['feature7']
test_dataset = test_dataset.merge(feature8, on='user_id', how='left')
test_dataset = test_dataset.merge(feature9, on='merchant_id', how='left')
test_dataset = test_dataset.merge(feature10, on='coupon_id', how='left')
test_dataset['feature11_week'] = feature11['week']
test_dataset['feature11_one'] = feature11['one']
test_dataset['feature11_two'] = feature11['two']
test_dataset['feature11_three'] = feature11['three']
test_dataset['feature11_four'] = feature11['four']
test_dataset['feature11_five'] = feature11['five']
test_dataset['feature11_six'] = feature11['six']
test_dataset['feature11_seven'] = feature11['seven']
test_dataset['feature11_eight'] = feature11['eight']
test_dataset = test_dataset.merge(feature13, on='user_id', how='left')
#########################################################
train_data = label_data.iloc[:, 4:-1]
label_data = label_data.iloc[:, 3]
test_data = test_dataset.iloc[:, 3:-1]

mode = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, subsample=0.7)
mode.fit(train_data, label_data)
test_pre = mode.predict_proba(test_data)[:, 1]
test_pre = pd.DataFrame(test_pre)
test_pre.columns = ['probability']

test_dataset = test_dataset_two.iloc[:, [0, 2, 5]]
test_dataset['probability'] = test_pre['probability']
test_dataset.to_csv('sample_submission.csv', index=None, header=None)
print test_dataset
