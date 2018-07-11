#coding=utf-8

import time
import pandas as pd
import numpy as np
import xgboost as xgb

start_time = time.clock()


# ijcai_train_data = pd.read_csv(r'round2_train.txt', engine='c', nrows=100, sep=' ', usecols=['instance_id', 'user_gender_id', 'user_occupation_id', 'item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
#                     'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level', 'context_timestamp', 'is_trade'])
#
# ijcai_test_data = pd.read_csv(r'ijcai_test_data.csv', engine='c', usecols=['instance_id', 'user_gender_id', 'user_occupation_id', 'item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
#                     'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level', 'context_timestamp'])
#
# print('数据读取完毕。。。')
#
# '''show records time'''
# ijcai_train_data['context_timestamp'] = [time.localtime(index) for index in ijcai_train_data['context_timestamp']]   # 训练集时间戳处理 #
# ijcai_train_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_train_data['context_timestamp']]
# ijcai_train_data['timestamp_hour'] = [int(time.strftime("%H", index)) for index in ijcai_train_data['context_timestamp']]
#
# ijcai_test_data['context_timestamp'] = [time.localtime(index) for index in ijcai_test_data['context_timestamp']]    # 测试集时间戳处理 #
# ijcai_test_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_test_data['context_timestamp']]
# ijcai_test_data['timestamp_hour'] = [int(time.strftime("%H", index)) for index in ijcai_test_data['context_timestamp']]
#
# ijcai_train_data.loc[:, 'user_age_level'] = [index - 1000 if index != -1 else index for index in ijcai_train_data['user_age_level']]
# ijcai_train_data.loc[:, 'user_occupation_id'] = [index - 2000 if index != -1 else index for index in ijcai_train_data['user_occupation_id']]
# ijcai_train_data.loc[:, 'user_star_level'] = [index - 3000 if index != -1 else index for index in ijcai_train_data['user_star_level']]
# ijcai_train_data.loc[:, 'shop_star_level'] = [index - 5000 if index != -1 else index for index in ijcai_train_data['shop_star_level']]
#
# print(ijcai_train_data.info())
# print(ijcai_train_data.describe())
# exit(0)
#
# # print(ijcai_train_data)
#
# # ijcai_train_morning = ijcai_train_data[ijcai_train_data['timestamp_hour'] < 12]
# # ijcai_train_afternoon = ijcai_train_data[ijcai_train_data['timestamp_hour'] >= 12]
# # print(ijcai_train_morning)
# # print(ijcai_train_afternoon)
# # del ijcai_train_data     # release train_data memory #
#
#
# train_data_one = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180831']
# train_data_two = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180901']
# train_data_three = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180902']
# train_data_four = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180903']
# train_data_five = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180904']
# train_data_six = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180905']
# train_data_seven = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180906']
# del ijcai_train_data     # release train_data memory #
#
# # print(train_data_one)
# # print(train_data_two)
# # print(train_data_three)
# # print(train_data_four)
# # print(train_data_five)
# # print(train_data_six)
# # print(train_data_seven)
#
# print('数据预处理完毕。。。')


# ijcai_test_data = pd.read_csv(r'ijcai_test_data.csv', engine='c', usecols=['instance_id', 'user_gender_id', 'user_occupation_id', 'item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
#                     'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level', 'context_timestamp','context_page_id', 'item_id', 'user_id', 'shop_id', 'item_brand_id', 'item_city_id'])

# print(ijcai_test_data['item_brand_id'].value_counts().reset_index())
# print(ijcai_test_data['item_city_id'].value_counts().reset_index())

'''display time'''
# ijcai_test_data = pd.read_csv(r'round2_ijcai_18_test_b_20180510.txt', engine='c',  sep=' ', nrows=None, usecols=['instance_id', 'context_timestamp'])
# ijcai_test_data['context_timestamp'] = [time.localtime(index) for index in ijcai_test_data['context_timestamp']]    # 测试集时间戳处理 #
# ijcai_test_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_test_data['context_timestamp']]
# ijcai_test_data['timestamp_hour'] = [int(time.strftime("%H", index)) for index in ijcai_test_data['context_timestamp']]
#
# # print(ijcai_test_data[['context_timestamp', 'timestamp_year_mouth_day', 'timestamp_hour']])
#
# print(ijcai_test_data.shape)
# print(ijcai_test_data.drop_duplicates(['instance_id']).shape)
# print(ijcai_test_data[ijcai_test_data['timestamp_year_mouth_day'] != 20180907].shape)
# print(ijcai_test_data[ijcai_test_data['timestamp_hour'] < 12].shape)


# ijcai_test_data_a = pd.read_csv(r'ijcai_test_data.csv', engine='c', usecols=['instance_id'])
ijcai_test_data_b = pd.read_csv(r'round2_ijcai_18_test_b_20180510.txt', sep=' ', engine='c', usecols=['instance_id', 'item_id', 'user_id'])

# print(pd.Series(list(set(ijcai_test_data_a['instance_id']).union(set(ijcai_test_data_b['instance_id'])))))

print(ijcai_test_data_b.shape)
print(ijcai_test_data_b.drop_duplicates(['item_id', 'user_id']).shape)


