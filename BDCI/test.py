
#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# train_data-ccf_first_round_user_shop_behavior.csv
# [1138015 rows x 11 columns]

# ABtest-evaluation_public.csv
# [483931 rows x 7 columns]

# train_data-ccf_first_round_shop_info.csv
# [8477 rows x 6 columns]

# 正负样本比例 #
# 共有97家商场 #
# 共8477家商铺，每个不同商场不存在相同商铺 #
# 用户行为日期2017-08-01 至 2017-08-31 #

# def shop_list(group):
#     return set(group)
#
# def call_signal(group):
#     return list(group)

read_data_one = pd.read_csv(r'D:\aliyun_ccf\train_data-ccf_first_round_user_shop_behavior.csv', header=None, low_memory=False) #, nrows=170000 #
read_data_one.columns = ['user_id', 'shop_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos']

# read_data_two = pd.read_csv(r'D:\aliyun_ccf\ABtest-evaluation_public.csv', header=None, low_memory=False)
# read_data_two.columns = ['row_id', 'user_id', 'mall_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos']

# read_data_three = pd.read_csv(r'D:\aliyun_ccf\train_data-ccf_first_round_shop_info.csv', header=None, low_memory=False)
# read_data_three.columns = ['shop_id', 'category_id', 'longitude', 'latitude', 'price', 'mall_id']


# read_data_one = read_data_one.iloc[:, [0, 1]]# 提取user_id shop_id #
# read_data_three = read_data_three.iloc[:, [0, 5]] # 提取shop_id mall_id #

print read_data_one
# print read_data_two
# print read_data_three
# print pd.pivot_table(read_data_three, index='mall_id', values='shop_id', aggfunc=shop_list)  # 统计每个商场有哪些商铺 #
# read_data_one = read_data_one.merge(read_data_three, on='shop_id', how='left')
# print read_data_one
# print read_data_one.merge(read_data_three, on='mall_id', how='left')
# print read_data_one[read_data_one['shop_id_x'] == read_data_one['shop_id_y']]

# read_data_one = read_data_one.merge(read_data_three, on='shop_id', how='left')
# print read_data_one

# print read_data_three['shop_id'].drop_duplicates()
# print read_data_one[read_data_one['mall_id'].notnull()].count()

# 用户消费行为table和商铺信息table融合 index=shop_id  融合后的表和商铺信息表融合index=mall_id
# ABtable和商铺信息table融合 index=mall_id
def deal(singal):  # 计算wifi信号强度 #
    str = singal.split(';')
    List = []
    for index in str:
        List.append(index.split('|')[1])
    return List
# read_data_two['number'] = [deal(index) for index in read_data_two['wifi_infos']]

# read_data_two['number'] = [len(index.split(';')) for index in read_data_two['wifi_infos']]
# print read_data_two

read_data_one['date'] = [index.split(' ')[0] for index in read_data_one['time_stamp']]
read_data_one['date'] = [int(index.replace('-', '')) for index in read_data_one['date']]

# print read_data_one['date'].max()
# print read_data_one['date'].min()
feature = pd.pivot_table(read_data_one, index='date', values='user_id', aggfunc='count')
print feature.index
# print read_data_one
# print feature
plt.plot(feature)
plt.show()
