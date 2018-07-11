#coding=utf-8

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_time = time.clock()


'''get timestamp set'''
# ijcai_train_data = pd.read_csv(r'round2_train.txt', sep=' ', usecols=['context_timestamp'])
# ijcai_test_data = pd.read_csv(r'ijcai_test_data.csv', usecols=['context_timestamp'])
# ijcai_test_data_b = pd.read_csv(r'round2_ijcai_18_test_b_20180510.txt', sep=' ', engine='c', usecols=['context_timestamp'])
# ijcai_test_data = pd.concat([ijcai_test_data, ijcai_test_data_b])
# print('数据读取完毕。。。')
# ijcai_train_data['context_timestamp'] = [time.localtime(index) for index in ijcai_train_data['context_timestamp']]   # 训练集时间戳处理 #
# ijcai_train_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d%H", index) for index in ijcai_train_data['context_timestamp']]
# ijcai_train_data['timestamp_year_mouth_day'].value_counts().reset_index().sort_values(by=['index']).to_csv('train_time.csv', index=None)
# # # ijcai_train_data['hour'] = [time.strftime("%H", index) for index in ijcai_train_data['context_timestamp']]
# # # ijcai_train_data['timestamp_hour_minute_second'] = [time.strftime("%H:%M:%S", index) for index in ijcai_train_data['context_timestamp']]
#
# ijcai_test_data['context_timestamp'] = [time.localtime(index) for index in ijcai_test_data['context_timestamp']]    # 测试集时间戳处理 #
# ijcai_test_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d%H", index) for index in ijcai_test_data['context_timestamp']]
# ijcai_test_data['timestamp_year_mouth_day'].value_counts().reset_index().sort_values(by=['index']).to_csv('test_time.csv', index=None)
#
# # ijcai_test_data['hour'] = [time.strftime("%H", index) for index in ijcai_test_data['context_timestamp']]
# # ijcai_test_data['timestamp_hour_minute_second'] = [time.strftime("%H:%M:%S", index) for index in ijcai_test_data['context_timestamp']]
#
# # print(ijcai_train_data['timestamp_year_mouth_day'].value_counts().reset_index().sort_values(by=['index']))
# # print(ijcai_test_data['timestamp_year_mouth_day'].value_counts().reset_index().sort_values(by=['index']))


'''show timestamp distribute'''
# train_time = pd.read_csv('train_time.csv', low_memory=False)
# test_time = pd.read_csv('test_time.csv', low_memory=False)
# train_time = train_time.append(test_time)
# # train_time = train_time[train_time['index'] >= 2018090500]
# # print(train_time)
# # print(test_time)
# x = range(len(list(train_time['timestamp_year_mouth_day'])))
# y = list(train_time['timestamp_year_mouth_day'])
#
# plt.bar(x, y, color='blue')
# plt.xlabel('Time')
# plt.ylabel('Records')
# plt.title('Time-Records')
# plt.xticks(x, list(train_time['index']), rotation=90, fontproperties='SimHei')
# plt.show()


'''item_category_list analyse'''
# # '1909641874861640857', '453525480366550911', '7226013370341271704', '6693726201323251689', '8769426218101861255', '7314150500379498593', '6670526099037031245', '3434689896486063330', '6254910033820815248', '1852600517265062354', '8468007938333142982', '926205401302902289', '5685690139879409547', '7423553047267511438', '2211060154630359130', '3089254302947620489', '768579787521575291', '5066527928272238333', '2871729383671301763', '1367177154073382718', '1147074168968532252', '8009556227083201357', '4911723539855588624', '1920084168104334820', '8841625760168847319', '597424223456586363', '1916390345133212703', '8468370105908620354', '3613783563199627217', '3348197449185791127', '394915394741096735'
# ijcai_train_data = pd.read_csv(r'round2_train.txt', engine='c', nrows=10000, sep=' ', usecols=['item_category_list'])    # 10000
# ijcai_train_data.loc[:, 'item_category_list_str'] = [index.split(';') for index in ijcai_train_data['item_category_list']]  # 按;切片 #
# ijcai_train_data.loc[:, 'item_category_list_len'] = [len(index) for index in ijcai_train_data['item_category_list_str']]  # 切片后字符串长度 #
#
# # ijcai_train_data.loc[:, 'item_category_list_two'] = [index[1] for index in ijcai_train_data['item_category_list_str']]
# ijcai_train_data.loc[:, 'item_category_list_two'] = ijcai_train_data[['item_category_list_str', 'item_category_list_len']].apply(lambda x: x.item_category_list_str[2] if x.item_category_list_len == 3 else x.item_category_list_str[1], axis=1)
#
# print(set(ijcai_train_data['item_category_list_two']))
# print(ijcai_train_data)
#
# print()
# print()
# print()
# print('This program spend time:', time.clock() - start_time, ' s')
# print('This program spend time:', (time.clock() - start_time)/60, ' min')

'''instance_id & context_id check'''
# ijcai_train_data = pd.read_csv(r'round2_train.txt', engine='c', nrows=None,  sep=' ', usecols=['instance_id', 'context_id'])
# print(ijcai_train_data.shape)
# print(ijcai_train_data.drop_duplicates(['instance_id']).shape)


# attributes = ['instance_id', 'user_gender_id', 'user_occupation_id', 'item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level', 'item_category_list', 'context_timestamp']
# print(attributes)
# attributes.append('is_trade')
# print(attributes)
# print('Hello')


