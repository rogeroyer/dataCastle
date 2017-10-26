#coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

# read_data = pd.read_csv(r'D:\aliyun\ccf_offline_stage1_train.csv', header=None)
# read_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
#
# # 划分数据集 #
# read_data = read_data[read_data['date_received'] != 'null']
# # train_data = read_data[((read_data['date_received'] >= '20160401') & (read_data['date_received'] < '20160615')) | (read_data['date_received'] == 'null')]
# train_data = read_data[(read_data['date_received'] >= '20160315') & (read_data['date_received'] < '20160615')]
# # print train_data
# label_data = read_data[(read_data['date_received'] >= '20160515') & (read_data['date_received'] < '20160615')]
# # 给未消费数据打标为0 #
# label_data_part1 = label_data[label_data['date'] == 'null'] #318750
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
# train_data.to_csv('train_data.csv', index=None, header=None)
# label_data.to_csv('label_data.csv', index=None, header=None)
# # print train_data
# # print label_data

####################################################################
####################################################################
####################################################################
train_data = pd.read_csv(r'train_data.csv', header=None)
train_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
train_data['date_received'] = [str(index) for index in train_data['date_received']]

###############################################################################
# Just for test #
# 该用户领了多少商家的优惠券 #
# def call_mode(group):
#     return group #.mode()
# 求众数的函数 #
def call_mode(group):
    d = {}
    s = set()
    for x in group:
        if x in s:
            d[x] += 1;
        else:
            s.add(x)
            d[x] = 1
    for key in d:
        if d[key] == max(d.values()):
            return key


# feature1 = pd.pivot_table(train_data, index=['user_id', 'merchant_id'], values='coupon_id', aggfunc='count')
# # feature2 = pd.pivot_table(train_data, index='user_id', values='coupon_id', aggfunc='count')
# print feature1
# feature1.to_csv('test.csv', header=None)
# feature1['user_id', 'merchant_id'] = feature1.index[0]
# print feature1
# print feature2
# print train_data.groupby('user_id')['coupon_id'].count()

# 求set数目的aggfunc函数 #
def call_set(group):
    return len(set(group))

# 每个商户发的优惠券被多少个不同用户领取并归一化 #
feature22 = pd.pivot_table(train_data, index='merchant_id', values='user_id', aggfunc=call_set)
feature22['merchant_id'] = feature22.index
feature22.columns = ['feature22', 'merchant_id']
m = feature22['feature22'].max()
n = feature22['feature22'].min()
feature22['feature22_one'] = [1.0*(index-n)/(m-n) for index in feature22['feature22']]
# print feature22

# 每种优惠券被多少个不同用户领取并归一化 #
feature23 = pd.pivot_table(train_data, index='coupon_id', values='user_id', aggfunc=call_set)
feature23['merchant_id'] = feature23.index
feature23.columns = ['feature23', 'coupon_id']
m = feature23['feature23'].max()
n = feature23['feature23'].min()
feature23['feature23_one'] = [1.0*(index-n)/(m-n) for index in feature23['feature23']]
# print feature23

# 每个商户发了多少优惠券 #
feature1 = pd.pivot_table(train_data, index='merchant_id', values='coupon_id', aggfunc='count')
feature1['merchant_id'] = feature1.index
feature1.columns = ['feature1', 'merchant_id']

# 每位用户领了多少张优惠券 #
feature2 = pd.pivot_table(train_data[train_data['coupon_id'].notnull()], index='user_id', values='coupon_id', aggfunc='count')
feature2['user_id'] = feature2.index
feature2.columns = ['feature2', 'user_id']

# 该用户领了多少商家的优惠券 #
feature14 = pd.pivot_table(train_data, index='user_id', values='merchant_id', aggfunc=call_set)
feature14['user_id'] = feature14.index
feature14.columns = ['feature14', 'user_id']
# print feature14

# 每个用户领了多少不同类型的优惠券 #
def call_set(group):
    return len(set(group))
feature15 = pd.pivot_table(train_data, index='user_id', values='coupon_id', aggfunc=call_set)
feature15['user_id'] = feature15.index
feature15.columns = ['feature15', 'user_id']
# print feature15

# 每个商户发行了多少不同类型的优惠券 #
feature21 = pd.pivot_table(train_data, index='merchant_id', values='coupon_id', aggfunc=call_set)
feature21['merchant_id'] = feature21.index
feature21.columns = ['feature21', 'merchant_id']
# print feature21

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
feature = pd.pivot_table(train_data, index='user_id', values='distance', aggfunc=call_mode)
feature3['four0'] = feature['distance']
# print feature3
#
# 用户距离 3个 # # 均值，中位数，最大，最小 再两两交叉加减乘，本身与本身加减乘#
feature19 = pd.pivot_table(train_data, index='merchant_id', values='distance', aggfunc='mean')
feature19['merchant_id'] = feature19.index
feature19.columns = ['feature19', 'merchant_id']
feature = pd.pivot_table(train_data, index='merchant_id', values='distance', aggfunc='max')
feature19['one'] = feature['distance']
feature = pd.pivot_table(train_data, index='merchant_id', values='distance', aggfunc='min')
feature19['two'] = feature['distance']
feature = pd.pivot_table(train_data, index='merchant_id', values='distance', aggfunc='median')
feature19['three'] = feature['distance']
feature = pd.pivot_table(train_data, index='merchant_id', values='distance', aggfunc=call_mode)
feature19['four0'] = feature['distance']
feature19['four'] = feature19['feature19'] + feature19['one']
feature19['four1'] = feature19['feature19'] - feature19['one']
feature19['four2'] = feature19['feature19'] * feature19['one']
feature19['five'] = feature19['feature19'] + feature19['two']
feature19['five1'] = feature19['feature19'] - feature19['two']
feature19['five2'] = feature19['feature19'] * feature19['two']
feature19['six'] = feature19['feature19'] + feature19['three']
feature19['six1'] = feature19['feature19'] - feature19['three']
feature19['six2'] = feature19['feature19'] * feature19['three']
feature19['four3'] = feature19['feature19'] + feature19['four0']
feature19['four4'] = feature19['feature19'] - feature19['four0']
feature19['four5'] = feature19['feature19'] * feature19['four0']
feature19['seven'] = feature19['one'] + feature19['two']
feature19['seven1'] = feature19['one'] - feature19['two']
feature19['seven2'] = feature19['one'] * feature19['two']
feature19['eight'] = feature19['one'] + feature19['three']
feature19['eight1'] = feature19['one'] - feature19['three']
feature19['eight2'] = feature19['one'] * feature19['three']
feature19['eight3'] = feature19['one'] + feature19['four0']
feature19['eight4'] = feature19['one'] - feature19['four0']
feature19['eight5'] = feature19['one'] * feature19['four0']
feature19['nine'] = feature19['two'] + feature19['three']
feature19['nine1'] = feature19['two'] - feature19['three']
feature19['nine2'] = feature19['two'] * feature19['three']
feature19['nine3'] = feature19['two'] + feature19['four0']
feature19['nine4'] = feature19['two'] - feature19['four0']
feature19['nine5'] = feature19['two'] * feature19['four0']
feature19['nine6'] = feature19['three'] + feature19['four0']
feature19['nine7'] = feature19['three'] - feature19['four0']
feature19['nine8'] = feature19['three'] * feature19['four0']
feature19['ten'] = feature19['one'] + feature19['one']
feature19['ten2'] = feature19['one'] * feature19['one']
feature19['eleven'] = feature19['feature19'] + feature19['feature19']
feature19['eleven2'] = feature19['feature19'] * feature19['feature19']
feature19['twelve'] = feature19['two'] + feature19['two']
feature19['twelve2'] = feature19['two'] * feature19['two']
feature19['thirteen'] = feature19['three'] + feature19['three']
feature19['thirteen2'] = feature19['three'] * feature19['three']
feature19['thirteen3'] = feature19['four0'] + feature19['four0']
feature19['thirteen4'] = feature19['four0'] * feature19['four0']
# print feature3
# print feature19

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
# 从打标的数据集中提取特征 #
# 读取已经打标的数据集 #
label_data = pd.read_csv('label_data.csv', header=None)
label_data.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date', 'label']
# print label_data
# 用户领取特定商家的优惠券张数 #
feature20 = pd.pivot_table(label_data, index=['user_id', 'merchant_id'], values='coupon_id', aggfunc='count')
feature20.to_csv('test.csv', header=None)
feature20 = pd.read_csv(r'test.csv', header=None)
feature20.columns = ['user_id', 'merchant_id', 'feature20']

# 优惠率 #
feature4 = label_data.iloc[:, [2, 3]]
feature4['discount_rate'] = [index if index[0] == '0' else (1 - (int(str(index).split(':')[1])*1.0)/int(str(index).split(':')[0]))
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
# 满多少归一化 #
feature17 = feature5
m = feature17['up'].max()
n = feature17['up'].min()
feature17['up'] = [1.0*(index-n)/(m-n) for index in feature17['up']]
# print feature17

# 减多少 #
feature6 = label_data.iloc[:, [2, 3]]
feature6.columns = ['coupon_id', 'down']
feature6['down'] = [index if index[0] == '0' else int(str(index).split(':')[1]) for index in feature6['down']]
up_median = feature6['down'].median()
feature6['down'] = [up_median if str(index)[0] == '0' else index for index in feature6['down']]
# print feature6
# 减多少归一化 #
feature18 = feature6
m = feature18['down'].max()
n = feature18['down'].min()
feature18['down'] = [1.0*(index-n)/(m-n) for index in feature18['down']]
# print feature18

# 折扣类型 #
feature7 = label_data.iloc[:, [0, 2, 3]] #
feature7.columns = ['user_id', 'coupon_id', 'feature7']
feature7['feature7'] = [0 if index[0] == '0' else 1 for index in feature7['feature7']]
# print feature7

# 每个用户喜欢领取的优惠券类型 #
feature16 = pd.pivot_table(feature7, index='user_id', values='feature7', aggfunc=call_mode)
feature16['user_id'] = feature16.index
feature16.columns = ['feature16', 'user_id']
# print feature16

# # 领券日期相关特征 9个 #
label_data['date_received'] = pd.to_datetime(train_data['date_received'])
label_data['week'] = [index.weekday()+1 for index in label_data['date_received']]
feature11 = label_data.iloc[:, [0, 8]]   ## 改 ##
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
# feature = pd.pivot_table(feature11, index='user_id', values='one', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='two', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='three', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='four', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='five', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='six', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='seven', aggfunc='sum')
# feature['user_id'] = feature.index
# # print feature
# feature11 = feature11.merge(feature, on='user_id', how='left')
# print feature11

# 判断是否只领了一次券 #
feature = label_data.iloc[:, [0, 5]]
feature13 = pd.pivot_table(feature, index='user_id', values='date_received', aggfunc='count')
feature13['user_id'] = feature13.index
feature13['feature13'] = [1 if index == 1 else 0 for index in feature13['date_received']]
feature13 = feature13.drop('date_received', axis=1)
# print feature13

###################################################################
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
# label_data = label_data.merge(feature11, on='user_id', how='left')
label_data = label_data.merge(feature13, on='user_id', how='left')
label_data = label_data.merge(feature14, on='user_id', how='left')
label_data = label_data.merge(feature15, on='user_id', how='left')
label_data = label_data.merge(feature16, on='user_id', how='left')
label_data['feature17'] = feature17['up']
label_data['feature18'] = feature18['down']
label_data = label_data.merge(feature19, on='merchant_id', how='left')
label_data = label_data.merge(feature20, on=['user_id', 'merchant_id'], how='left')
label_data = label_data.merge(feature21, on='merchant_id', how='left')
label_data = label_data.merge(feature22, on='merchant_id', how='left')
label_data = label_data.merge(feature23, on='coupon_id', how='left')

print label_data


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
#
# 每个商户发的优惠券被多少个不同用户领取并归一化 #
feature22 = pd.pivot_table(test_dataset, index='merchant_id', values='user_id', aggfunc=call_set)
feature22['merchant_id'] = feature22.index
feature22.columns = ['feature22', 'merchant_id']
m = feature22['feature22'].max()
n = feature22['feature22'].min()
feature22['feature22_one'] = [1.0*(index-n)/(m-n) for index in feature22['feature22']]
# print feature22

# 每种优惠券被多少个不同用户领取并归一化 #
feature23 = pd.pivot_table(test_dataset, index='coupon_id', values='user_id', aggfunc=call_set)
feature23['merchant_id'] = feature23.index
feature23.columns = ['feature23', 'coupon_id']
m = feature23['feature23'].max()
n = feature23['feature23'].min()
feature23['feature23_one'] = [1.0*(index-n)/(m-n) for index in feature23['feature23']]
# print feature23

# 用户领取特定商家的优惠券张数 #
feature20 = pd.pivot_table(test_dataset, index=['user_id', 'merchant_id'], values='coupon_id', aggfunc='count')
feature20.to_csv('test.csv', header=None)
feature20 = pd.read_csv(r'test.csv', header=None)
feature20.columns = ['user_id', 'merchant_id', 'feature20']

# 每个商户发了多少优惠券 #
feature1 = pd.pivot_table(test_dataset, index='merchant_id', values='coupon_id', aggfunc='count')
feature1['merchant_id'] = feature1.index
feature1.columns = ['feature1', 'merchant_id']

# 每位用户领了多少张优惠券 #
feature2 = pd.pivot_table(test_dataset[test_dataset['coupon_id'].notnull()], index='user_id', values='coupon_id', aggfunc='count')
feature2['user_id'] = feature2.index
feature2.columns = ['feature2', 'user_id']

# 该用户领了多少商家的优惠券 #
def call_set(group):
    return len(set(group))
feature14 = pd.pivot_table(test_dataset, index='user_id', values='merchant_id', aggfunc=call_set)
feature14['user_id'] = feature14.index
feature14.columns = ['feature14', 'user_id']
# print feature14

# 每个用户领了多少不同类型的优惠券 #
feature15 = pd.pivot_table(test_dataset, index='user_id', values='coupon_id', aggfunc=call_set)
feature15['user_id'] = feature15.index
feature15.columns = ['feature15', 'user_id']
# print feature15

# 每个商户发行了多少不同类型的优惠券 #
feature21 = pd.pivot_table(test_dataset, index='merchant_id', values='coupon_id', aggfunc=call_set)
feature21['merchant_id'] = feature21.index
feature21.columns = ['feature21', 'merchant_id']
# print feature21

# 商户距离 4个 #
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
feature = pd.pivot_table(test_dataset, index='user_id', values='distance', aggfunc=call_mode)
feature3['four0'] = feature['distance']

# 用户距离 3个 # # 均值，中位数，最大，最小 再两两交叉加减乘，本身与本身加减乘#
feature19 = pd.pivot_table(test_dataset, index='merchant_id', values='distance', aggfunc='mean')
feature19['merchant_id'] = feature19.index
feature19.columns = ['feature19', 'merchant_id']
feature = pd.pivot_table(test_dataset, index='merchant_id', values='distance', aggfunc='max')
feature19['one'] = feature['distance']
feature = pd.pivot_table(test_dataset, index='merchant_id', values='distance', aggfunc='min')
feature19['two'] = feature['distance']
feature = pd.pivot_table(test_dataset, index='merchant_id', values='distance', aggfunc='median')
feature19['three'] = feature['distance']
feature = pd.pivot_table(test_dataset, index='merchant_id', values='distance', aggfunc=call_mode)
feature19['four0'] = feature['distance']
feature19['four'] = feature19['feature19'] + feature19['one']
feature19['four1'] = feature19['feature19'] - feature19['one']
feature19['four2'] = feature19['feature19'] * feature19['one']
feature19['five'] = feature19['feature19'] + feature19['two']
feature19['five1'] = feature19['feature19'] - feature19['two']
feature19['five2'] = feature19['feature19'] * feature19['two']
feature19['six'] = feature19['feature19'] + feature19['three']
feature19['six1'] = feature19['feature19'] - feature19['three']
feature19['six2'] = feature19['feature19'] * feature19['three']
feature19['four3'] = feature19['feature19'] + feature19['four0']
feature19['four4'] = feature19['feature19'] - feature19['four0']
feature19['four5'] = feature19['feature19'] * feature19['four0']
feature19['seven'] = feature19['one'] + feature19['two']
feature19['seven1'] = feature19['one'] - feature19['two']
feature19['seven2'] = feature19['one'] * feature19['two']
feature19['eight'] = feature19['one'] + feature19['three']
feature19['eight1'] = feature19['one'] - feature19['three']
feature19['eight2'] = feature19['one'] * feature19['three']
feature19['eight3'] = feature19['one'] + feature19['four0']
feature19['eight4'] = feature19['one'] - feature19['four0']
feature19['eight5'] = feature19['one'] * feature19['four0']
feature19['nine'] = feature19['two'] + feature19['three']
feature19['nine1'] = feature19['two'] - feature19['three']
feature19['nine2'] = feature19['two'] * feature19['three']
feature19['nine3'] = feature19['two'] + feature19['four0']
feature19['nine4'] = feature19['two'] - feature19['four0']
feature19['nine5'] = feature19['two'] * feature19['four0']
feature19['nine6'] = feature19['three'] + feature19['four0']
feature19['nine7'] = feature19['three'] - feature19['four0']
feature19['nine8'] = feature19['three'] * feature19['four0']
feature19['ten'] = feature19['one'] + feature19['one']
feature19['ten2'] = feature19['one'] * feature19['one']
feature19['eleven'] = feature19['feature19'] + feature19['feature19']
feature19['eleven2'] = feature19['feature19'] * feature19['feature19']
feature19['twelve'] = feature19['two'] + feature19['two']
feature19['twelve2'] = feature19['two'] * feature19['two']
feature19['thirteen'] = feature19['three'] + feature19['three']
feature19['thirteen2'] = feature19['three'] * feature19['three']
feature19['thirteen3'] = feature19['four0'] + feature19['four0']
feature19['thirteen4'] = feature19['four0'] * feature19['four0']
# print feature3
# print feature19

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
feature4['discount_rate'] = [index if index[0] == '0' else (1 - (int(str(index).split(':')[1])*1.0)/int(str(index).split(':')[0]))
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
# 满多少归一化 #
feature17 = feature5
m = feature17['up'].max()
n = feature17['up'].min()
feature17['up'] = [1.0*(index-n)/(m-n) for index in feature17['up']]
# print feature17

# 减多少 #
feature6 = test_dataset.iloc[:, [2, 3]]
feature6.columns = ['coupon_id', 'down']
feature6['down'] = [index if index[0] == '0' else int(str(index).split(':')[1]) for index in feature6['down']]
up_median = feature6['down'].median()
feature6['down'] = [up_median if str(index)[0] == '0' else index for index in feature6['down']]
# print feature6

# 减多少归一化 #
feature18 = feature6
m = feature18['down'].max()
n = feature18['down'].min()
feature18['down'] = [1.0*(index-n)/(m-n) for index in feature18['down']]
# print feature18

# 折扣类型 #
feature7 = test_dataset.iloc[:, [0, 2, 3]]
feature7.columns = ['user_id', 'coupon_id', 'feature7']
feature7['feature7'] = [0 if index[0] == '0' else 1 for index in feature7['feature7']]
# print feature7

# 每个用户喜欢领取的优惠券类型 #
feature16 = pd.pivot_table(feature7, index='user_id', values='feature7', aggfunc=call_mode)
feature16['user_id'] = feature16.index
feature16.columns = ['feature16', 'user_id']
# print feature16

# 领券日期相关特征 9个 #
test_dataset['date_received'] = [str(index) for index in test_dataset['date_received']]
test_dataset['date_received'] = pd.to_datetime(test_dataset['date_received'])
test_dataset['week'] = [index.weekday()+1 for index in test_dataset['date_received']]
feature11 = test_dataset.iloc[:, [0, 6]]
feature11['one'] = [1 if index == 1 else 0 for index in feature11['week']]
feature11['two'] = [1 if index == 2 else 0 for index in feature11['week']]
feature11['three'] = [1 if index == 3 else 0 for index in feature11['week']]
feature11['four'] = [1 if index == 4 else 0 for index in feature11['week']]
feature11['five'] = [1 if index == 5 else 0 for index in feature11['week']]
feature11['six'] = [1 if index == 6 else 0 for index in feature11['week']]
feature11['seven'] = [1 if index == 7 else 0 for index in feature11['week']]
set_one = set([7, 6])
feature11['eight'] = [1 if index in set_one else 0 for index in feature11['week']]
# feature = pd.pivot_table(feature11, index='user_id', values='one', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')

# feature = pd.pivot_table(feature11, index='user_id', values='two', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='three', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='four', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='five', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='six', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
#
# feature = pd.pivot_table(feature11, index='user_id', values='seven', aggfunc='sum')
# feature['user_id'] = feature.index
# feature11 = feature11.merge(feature, on='user_id', how='left')
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
# test_dataset = test_dataset.merge(feature11, on='user_id', how='left')
test_dataset = test_dataset.merge(feature13, on='user_id', how='left')
test_dataset = test_dataset.merge(feature14, on='user_id', how='left')
test_dataset = test_dataset.merge(feature15, on='user_id', how='left')
test_dataset = test_dataset.merge(feature16, on='user_id', how='left')
test_dataset['feature17'] = feature17['up']
test_dataset['feature18'] = feature18['down']
test_dataset = test_dataset.merge(feature19, on='merchant_id', how='left')
test_dataset = test_dataset.merge(feature20, on=['user_id', 'merchant_id'], how='left')
test_dataset = test_dataset.merge(feature21, on='merchant_id', how='left')
test_dataset = test_dataset.merge(feature22, on='merchant_id', how='left')
test_dataset = test_dataset.merge(feature23, on='coupon_id', how='left')
# test_dataset = test_dataset.fillna(0)
print test_dataset


#########################################################
train_data = label_data.iloc[:, 4:-1]
label_data = label_data.iloc[:, 3]
test_data = test_dataset.iloc[:, 3:-1]
# print train_data
# print label_data
# print test_data

mode = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, subsample=0.7)
mode.fit(train_data, label_data)
test_pre = mode.predict_proba(test_data)[:, 1]
test_pre = pd.DataFrame(test_pre)
test_pre.columns = ['probability']

test_dataset = test_dataset_two.iloc[:, [0, 2, 5]]
test_dataset['probability'] = test_pre['probability']
test_dataset.to_csv('sample_submission.csv', index=None, header=None)
print test_dataset

