#coding=utf-8
'''
Author:Roger
date:2018-04-08
offline:0.0803853705124
online:my 1
module:xgboost
'''

import pandas as pd
import numpy as np
import time
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree
from xgboost import plot_importance
from xgboost import plot_tree
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectKBest    # 特征选择 #
from sklearn.feature_selection import chi2

# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


'''读取数据集'''
ijcai_train_data = pd.read_csv('../data_pre_deal/ijcai_train_data.csv', low_memory=False)    # , nrows=1000 #
ijcai_test_data = pd.read_csv('../data_pre_deal/ijcai_test_data.csv', low_memory=False)

ijcai_train_data = ijcai_train_data.drop_duplicates(['instance_id'])   # 数据集去重 #
ijcai_train_data['context_timestamp'] = [time.localtime(index) for index in ijcai_train_data['context_timestamp']]   # 训练集时间戳处理 #
ijcai_train_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_train_data['context_timestamp']]
ijcai_train_data['hour'] = [time.strftime("%H", index) for index in ijcai_train_data['context_timestamp']]
ijcai_train_data['timestamp_hour_minute_second'] = [time.strftime("%H:%M:%S", index) for index in ijcai_train_data['context_timestamp']]

ijcai_test_data['context_timestamp'] = [time.localtime(index) for index in ijcai_test_data['context_timestamp']]    # 测试集时间戳处理 #
ijcai_test_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_test_data['context_timestamp']]
ijcai_test_data['hour'] = [time.strftime("%H", index) for index in ijcai_test_data['context_timestamp']]
ijcai_test_data['timestamp_hour_minute_second'] = [time.strftime("%H:%M:%S", index) for index in ijcai_test_data['context_timestamp']]

'''去噪声值'''
ijcai_train_data['shop_review_positive_rate'] = [index if index != -1 else ijcai_train_data['shop_review_positive_rate'].mean() for index in ijcai_train_data['shop_review_positive_rate']]
ijcai_train_data['shop_score_service'] = [index if index != -1 else ijcai_train_data['shop_score_service'].mean() for index in ijcai_train_data['shop_score_service']]
ijcai_train_data['shop_score_delivery'] = [index if index != -1 else ijcai_train_data['shop_score_delivery'].mean() for index in ijcai_train_data['shop_score_delivery']]
ijcai_train_data['shop_score_description'] = [index if index != -1 else ijcai_train_data['shop_score_description'].mean() for index in ijcai_train_data['shop_score_description']]
ijcai_train_data['item_sales_level'] = [index if index != -1 else ijcai_train_data['item_sales_level'].median() for index in ijcai_train_data['item_sales_level']]
ijcai_train_data['user_star_level'] = [index if index != -1 else ijcai_train_data['user_star_level'].median() for index in ijcai_train_data['user_star_level']]

ijcai_test_data['shop_review_positive_rate'] = [index if index != -1 else ijcai_test_data['shop_review_positive_rate'].mean() for index in ijcai_test_data['shop_review_positive_rate']]
ijcai_test_data['shop_score_service'] = [index if index != -1 else ijcai_test_data['shop_score_service'].mean() for index in ijcai_test_data['shop_score_service']]
ijcai_test_data['shop_score_delivery'] = [index if index != -1 else ijcai_test_data['shop_score_delivery'].mean() for index in ijcai_test_data['shop_score_delivery']]
ijcai_test_data['shop_score_description'] = [index if index != -1 else ijcai_test_data['shop_score_description'].mean() for index in ijcai_test_data['shop_score_description']]
ijcai_test_data['item_sales_level'] = [index if index != -1 else ijcai_test_data['item_sales_level'].median() for index in ijcai_test_data['item_sales_level']]
ijcai_test_data['user_star_level'] = [index if index != -1 else ijcai_test_data['user_star_level'].median() for index in ijcai_test_data['user_star_level']]

user_gender_id = ijcai_train_data['user_gender_id'].drop_duplicates()

'''训练集'''
train_data_one = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180918') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180920')]
train_data_two = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180919') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180921')]
train_data_three = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180920') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180922')]
train_data_four = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180921') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180923')]
# train_data_five = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180922') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180923')]
'''验证集'''
validate_data = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180922') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180924')]
# '''测试集'''
test_data = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180923') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180924')]                # 可以尝试去23-25日的作为测试集 #
test_data = test_data.iloc[:, :-1].append(ijcai_test_data)                                              # 提取消费特征时需要去掉.iloc[:, :-1] #

'''训练集标签'''
train_label = pd.DataFrame()
train_label = train_label.append(train_data_one[train_data_one['timestamp_year_mouth_day'] == '20180920'].loc[:, ['is_trade']])
train_label = train_label.append(train_data_two[train_data_two['timestamp_year_mouth_day'] == '20180921'].loc[:, ['is_trade']])
train_label = train_label.append(train_data_three[train_data_three['timestamp_year_mouth_day'] == '20180922'].loc[:, ['is_trade']])
train_label = train_label.append(train_data_four[train_data_four['timestamp_year_mouth_day'] == '20180923'].loc[:, ['is_trade']])
# train_label = train_label.append(train_data_five[train_data_five['timestamp_year_mouth_day'] == '20180923'].loc[:, ['is_trade']])

'''验证集标签'''
validate_label = validate_data[validate_data['timestamp_year_mouth_day'] == '20180924'].loc[:, ['is_trade']]


def max_nin(data):
    '''取Series最大最小值'''
    return data.max(), data.min()


def extract_cross_feature(dataset, use_index, use_value, function, feature_name):
    '''提取交叉组合特征'''
    feature_temp = pd.pivot_table(dataset, index=use_index, values=use_value, aggfunc=function)
    feature_temp[use_index] = feature_temp.index
    feature_temp.columns = [feature_name, use_index]
    return feature_temp


def extract_feature(dataset):
    '''提取特征'''
    # global train_label
    date = dataset['timestamp_year_mouth_day'].max()
    dataset.loc[:, 'shop_star_level'] = [index - 4999 for index in dataset['shop_star_level']]
    dataset.loc[:, 'context_page_id'] = [index - 4000 for index in dataset['context_page_id']]
    dataset.loc[:, 'user_star_level'] = [index - 3000 for index in dataset['user_star_level']]
    dataset_one = dataset[dataset['timestamp_year_mouth_day'] == date]       # Label区间 #
    dataset_two = dataset[dataset['timestamp_year_mouth_day'] != date]       # Feature区间 #
    # train_label = train_label.append(dataset_one.loc[:, ['is_trade']])


    '''属性特征   Attention:使用dataset_one数据集  '''
    feature = dataset_one.loc[:, ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'shop_id', 'context_id', 'hour', 'user_occupation_id', 'item_price_level']]
    attribute = ['item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
                 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level']

    for index in attribute:
        feature[index] = dataset_one.loc[:, [index]]

    shop_star_level_max, shop_star_level_min = max_nin(feature['shop_star_level'])
    feature['shop_star_level'] = [(index - shop_star_level_min)/(shop_star_level_max - shop_star_level_min) for index in feature['shop_star_level']]
    feature.loc[:, 'user_age_level'] = [index - 1000 if index > 0 else 3 for index in feature['user_age_level']]

    '''item系列'''
    dataset_one.loc[:, 'item_category_list_str'] = [index.split(';') for index in dataset_one['item_category_list']]  # 按;切片 #
    dataset_one.loc[:, 'item_category_list_len'] = [len(index) for index in dataset_one['item_category_list_str']]  # 切片后字符串长度 #
    dataset_one.loc[:, 'item_category_list_two'] = [index[1] for index in dataset_one['item_category_list_str']]
    item_category_list_one_two = ['8277336076276184272', '5755694407684602296', '4879721024980945592', '2011981573061447208',
                                  '7258015885215914736', '509660095530134768', '8710739180200009128', '5799347067982556520',
                                  '2642175453151805566', '2436715285093487584', '3203673979138763595', '22731265849056483',
                                  '1968056100269760729']

    for index in item_category_list_one_two:
        feature[index] = [1 if (index == string) else 0 for string in dataset_one['item_category_list_two']]

    feature['item_category_list_len'] = [index for index in dataset_one['item_category_list_len']]         # 可去特征 #  ####
    feature['item_category_list_len'] = dataset_one.loc[:, ['item_category_list_len']]
    '''item_property_list'''
    feature['item_property_list_len'] = [index.count(';')+1 for index in dataset_one['item_property_list']]
    item_property_list_max, item_property_list_min = max_nin(feature['item_property_list_len'])
    feature['item_property_list_num'] = [(index - item_property_list_min) / (item_property_list_max - item_property_list_min) for index in feature['item_property_list_len']]

    '''性别特征'''
    for index in user_gender_id:
        feature[index] = [1 if (index == string) else 0 for string in dataset_one['user_gender_id']]

    '''职业特征'''
    occupation_id = [2005, 2002, 2003, 2004]
    for index in occupation_id:
        feature[index] = [1 if (index == string) else 0 for string in dataset_one['user_occupation_id']]

    def cal_conversion_rate(group):
        return list(group).count(1) / len(group)

    def cal_click_count(group):
        return list(group).count(1)

    '''转化率特征'''
    item_conversion = pd.pivot_table(dataset_two, index='item_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_id', 'is_trade': 'item_trade'})
    item_brand_conversion = pd.pivot_table(dataset_two, index='item_brand_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_brand_id', 'is_trade': 'item_brand_trade'})
    item_city_conversion = pd.pivot_table(dataset_two, index='item_city_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_city_id', 'is_trade': 'item_city_trade'})
    shop_conversion = pd.pivot_table(dataset_two, index='shop_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'shop_id', 'is_trade': 'shop_trade'})
    user_conversion = pd.pivot_table(dataset_two, index='user_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'user_id', 'is_trade': 'user_trade'})
    '''成交数'''
    item_conversion_click = pd.pivot_table(dataset_two, index='item_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'item_id', 'is_trade': 'item_conversion_click'})
    item_brand_conversion_click = pd.pivot_table(dataset_two, index='item_brand_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'item_brand_id', 'is_trade': 'item_brand_conversion_click'})
    item_city_conversion_click = pd.pivot_table(dataset_two, index='item_city_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'item_city_id', 'is_trade': 'item_city_trade_click'})
    shop_conversion_click = pd.pivot_table(dataset_two, index='shop_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'shop_id', 'is_trade': 'shop_trade_click'})
    user_conversion_click = pd.pivot_table(dataset_two, index='user_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'user_id', 'is_trade': 'user_trade_click'})

    user_shop_trade_count = pd.pivot_table(dataset_two, index=['user_id', 'shop_id'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_shop_trade_count'})
    user_item_trade_count = pd.pivot_table(dataset_two, index=['user_id', 'item_id'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_item_trade_count'})

    feature = feature.merge(item_conversion, on='item_id', how='left').fillna(0)
    feature = feature.merge(item_brand_conversion, on='item_brand_id', how='left').fillna(0)
    feature = feature.merge(item_city_conversion, on='item_city_id', how='left').fillna(0)
    feature = feature.merge(shop_conversion, on='shop_id', how='left').fillna(0)
    feature = feature.merge(user_conversion, on='user_id', how='left').fillna(0)

    feature = feature.merge(item_conversion_click, on='item_id', how='left').fillna(0)      # 0.0804517872261 #
    feature = feature.merge(item_brand_conversion_click, on='item_brand_id', how='left').fillna(0)
    feature = feature.merge(item_city_conversion_click, on='item_city_id', how='left').fillna(0)
    feature = feature.merge(shop_conversion_click, on='shop_id', how='left').fillna(0)
    feature = feature.merge(user_conversion_click, on='user_id', how='left').fillna(0)

    feature = feature.merge(user_shop_trade_count, on=['user_id', 'shop_id'], how='left').fillna(0)   ####
    feature = feature.merge(user_item_trade_count, on=['user_id', 'item_id'], how='left').fillna(0)   ####

    '''点击数'''
    user_click = extract_cross_feature(dataset_two, 'user_id', 'instance_id', 'count', 'user_click')
    shop_click = extract_cross_feature(dataset_two, 'shop_id', 'instance_id', 'count', 'shop_click')
    item_click = extract_cross_feature(dataset_two, 'item_id', 'instance_id', 'count', 'item_click')
    item_brand_click = extract_cross_feature(dataset_two, 'item_brand_id', 'instance_id', 'count', 'item_brand_click')
    item_city_click = extract_cross_feature(dataset_two, 'item_city_id', 'instance_id', 'count', 'item_city_click')
    # user_shop_count = extract_cross_feature(dataset_two, ['user_id', 'shop_id'], 'instance_id', 'user_shop_count')
    # user_item_count = extract_cross_feature(dataset_two, ['user_id', 'item_id'], 'instance_id', 'user_item_count')
    user_shop_count = pd.pivot_table(dataset_two, index=['user_id', 'shop_id'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_shop_count'})
    user_item_count = pd.pivot_table(dataset_two, index=['user_id', 'item_id'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_item_count'})

    feature = feature.merge(user_click, on='user_id', how='left').fillna(0)
    feature = feature.merge(shop_click, on='shop_id', how='left').fillna(0)
    feature = feature.merge(item_click, on='item_id', how='left').fillna(0)
    feature = feature.merge(item_brand_click, on='item_brand_id', how='left').fillna(0)
    feature = feature.merge(item_city_click, on='item_city_id', how='left').fillna(0)
    feature = feature.merge(user_shop_count, on=['user_id', 'shop_id'], how='left').fillna(0)    ####
    feature = feature.merge(user_item_count, on=['user_id', 'item_id'], how='left').fillna(0)    ####

    '''user_id shop_id item_id item_brand_id item_city_id 关于hour的点击数&成交量特征'''
    user_hour_count = pd.pivot_table(dataset_two, index=['user_id', 'hour'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_hour_count'})
    user_shop_hour_count = pd.pivot_table(dataset_two, index=['user_id', 'shop_id', 'hour'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_shop_hour_count'})
    user_item_hour_count = pd.pivot_table(dataset_two, index=['user_id', 'item_id', 'hour'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_item_hour_count'})
    shop_hour_count = pd.pivot_table(dataset_two, index=['shop_id', 'hour'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'shop_hour_count'})
    item_hour_count = pd.pivot_table(dataset_two, index=['item_id', 'hour'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'item_hour_count'})
    item_brand_hour_count = pd.pivot_table(dataset_two, index=['item_brand_id', 'hour'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'item_brand_hour_count'})
    item_city_hour_count = pd.pivot_table(dataset_two, index=['item_city_id', 'hour'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'item_city_hour_count'})

    user_hour_trade_count = pd.pivot_table(dataset_two, index=['user_id', 'hour'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_hour_trade_count'})
    user_shop_hour_trade_count = pd.pivot_table(dataset_two, index=['user_id', 'shop_id', 'hour'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_shop_hour_trade_count'})
    user_item_hour_trade_count = pd.pivot_table(dataset_two, index=['user_id', 'item_id', 'hour'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_item_hour_trade_count'})
    shop_hour_trade_count = pd.pivot_table(dataset_two, index=['shop_id', 'hour'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'shop_hour_trade_count'})
    item_hour_trade_count = pd.pivot_table(dataset_two, index=['item_id', 'hour'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'item_hour_trade_count'})
    item_brand_hour_trade_count = pd.pivot_table(dataset_two, index=['item_brand_id', 'hour'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'item_brand_hour_trade_count'})
    item_city_hour_trade_count = pd.pivot_table(dataset_two, index=['item_city_id', 'hour'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'item_city_hour_trade_count'})

    feature = feature.merge(user_hour_count, on=['user_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(user_shop_hour_count, on=['user_id', 'shop_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(user_item_hour_count, on=['user_id', 'item_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(shop_hour_count, on=['shop_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(item_hour_count, on=['item_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(item_brand_hour_count, on=['item_brand_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(item_city_hour_count, on=['item_city_id', 'hour'], how='left').fillna(0)
    # 0.0802847596899
    feature = feature.merge(user_hour_trade_count, on=['user_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(user_shop_hour_trade_count, on=['user_id', 'shop_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(user_item_hour_trade_count, on=['user_id', 'item_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(shop_hour_trade_count, on=['shop_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(item_hour_trade_count, on=['item_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(item_brand_hour_trade_count, on=['item_brand_id', 'hour'], how='left').fillna(0)
    feature = feature.merge(item_city_hour_trade_count, on=['item_city_id', 'hour'], how='left').fillna(0)
    # 0.0802906999587
    '''交叉组合统计特征'''
    dataset_two.loc[:, 'item_property_list_set'] = [index.count(';') + 1 for index in dataset_two['item_property_list']]
    '''user系列'''
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'item_price_level', 'mean', 'user_item_price_level2'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'item_property_list_set', 'mean', 'user_item_property_list_set1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'item_sales_level', 'mean', 'user_item_sales_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'item_collected_level', 'mean', 'user_item_collected_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'item_pv_level', 'mean', 'user_item_pv_level1'), on='user_id', how='left')
    ##
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'context_page_id', 'mean', 'user_context_page_id1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'shop_review_num_level', 'mean', 'user_shop_review_num_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'shop_review_positive_rate', 'mean', 'user_shop_review_positive_rate1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'shop_star_level', 'mean', 'user_shop_star_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'shop_score_service', 'mean', 'user_shop_score_service1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'shop_score_delivery', 'mean', 'user_shop_score_delivery1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'user_id', 'shop_score_description', 'mean', 'user_shop_score_description1'), on='user_id', how='left')

    '''shop系列'''
    ###
    feature = feature.merge(extract_cross_feature(dataset_two, 'shop_id', 'item_property_list_set', 'mean', 'shop_item_property_list_set1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'shop_id', 'item_price_level', 'mean', 'shop_item_price_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'shop_id', 'item_sales_level', 'mean', 'shop_item_sales_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'shop_id', 'item_collected_level', 'mean', 'shop_item_collected_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'shop_id', 'item_pv_level', 'mean', 'shop_item_pv_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'shop_id', 'user_star_level', 'mean', 'shop_user_star_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'shop_id', 'context_page_id', 'mean', 'shop_context_page_id1'), on='shop_id', how='left')

    '''item系列'''
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'user_star_level', 'mean', 'item_user_star_level1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'context_page_id', 'mean', 'item_context_page_id1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'shop_review_num_level', 'mean', 'item_shop_review_num_level1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'shop_review_positive_rate', 'mean', 'item_shop_review_positive_rate1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'shop_star_level', 'mean', 'item_shop_star_level1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'shop_score_service', 'mean', 'item_shop_score_service1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'shop_score_delivery', 'mean', 'item_shop_score_delivery1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(dataset_two, 'item_id', 'shop_score_description', 'mean', 'item_shop_score_description1'), on='item_id', how='left')

    click_data = dataset_one.loc[:, ['user_id', 'timestamp_hour_minute_second']]
    '''返回最大值'''
    def return_max_value(group):
        return max(group)

    '''排序特征'''
    click_data['click_rank'] = click_data.groupby('user_id')['timestamp_hour_minute_second'].rank(method='min', ascending=True)
    click_data['click_rank_reverse'] = click_data.groupby('user_id')['timestamp_hour_minute_second'].rank(method='min', ascending=False)
    feature['click_rank'] = [index for index in click_data['click_rank']]
    feature['click_rank_reverse'] = [index for index in click_data['click_rank_reverse']]
    click_rank_max = pd.pivot_table(feature, index='user_id', values='click_rank', aggfunc=return_max_value).reset_index().rename(columns={'index': 'user_id', 'click_rank': 'click_max_value'})
    feature = feature.merge(click_rank_max, on='user_id', how='left')
    feature['is_min'] = feature[['click_rank']].apply(lambda x: 1 if x.click_rank == 1else 0, axis=1)
    feature['is_max'] = feature[['click_rank', 'click_max_value']].apply(lambda x: 1 if x.click_rank == x.click_max_value else 0, axis=1)

    # ''' predict_category_property 属性特征 '''
    # dataset_one.loc[:, 'category_count'] = [index.count(':') for index in dataset_one['predict_category_property']]
    # dataset_one.loc[:, 'one_count'] = [index.count('-1') for index in dataset_one['predict_category_property']]
    # dataset_one.loc[:, 'two_count'] = [index.count('/') for index in dataset_one['predict_category_property']]
    # dataset_one.loc[:, 'property_count'] = dataset_one['category_count'] - dataset_one['one_count'] + dataset_one['two_count']
    # # dataset_one.loc[:, 'property_category_mean'] = dataset_one['property_count'] / dataset_one['category_count']
    # dataset_one.loc[:, 'property_category_mean'] = dataset_one[['property_count', 'category_count']].apply(lambda x: x.property_count / x.category_count, axis=1)
    # feature = feature.merge(dataset_one[['instance_id', 'category_count', 'property_count', 'property_category_mean']], on='instance_id', how='left')
    # # print(feature[['instance_id', 'category_count', 'property_count', 'property_category_mean']])
    # print(dataset_one[['instance_id', 'category_count', 'property_count', 'property_category_mean']].info())

    ''' 用户属性的点击数、购买量、转化率 '''
    user_star_level_count = pd.pivot_table(dataset_two, index='user_star_level', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_star_level_count'})
    user_star_level_buy_count = pd.pivot_table(dataset_two, index='user_star_level', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_star_level_buy_count'})
    user_star_level_count = user_star_level_count.merge(user_star_level_buy_count, on='user_star_level', how='left')
    user_star_level_count['conversion_rate1'] = user_star_level_count['user_star_level_buy_count'] / user_star_level_count['user_star_level_count']
    # user_star_level_count['user_star_level'] = [index - 3000 for index in user_star_level_count['user_star_level']]

    user_occupation_id_count = pd.pivot_table(dataset_two, index='user_occupation_id', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_occupation_id_count'})
    user_occupation_buy_count = pd.pivot_table(dataset_two, index='user_occupation_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_occupation_buy_count'})
    user_occupation_id_count = user_occupation_id_count.merge(user_occupation_buy_count, on='user_occupation_id', how='left')
    user_occupation_id_count['conversion_rate2'] = user_occupation_id_count['user_occupation_buy_count'] / user_occupation_id_count['user_occupation_id_count']

    user_age_level_count = pd.pivot_table(dataset_two, index='user_age_level', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_age_level_count'})
    user_age_level_buy_count = pd.pivot_table(dataset_two, index='user_age_level', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_age_level_buy_count'})
    user_age_level_count = user_age_level_count.merge(user_age_level_buy_count, on='user_age_level', how='left')
    user_age_level_count['conversion_rate3'] = user_age_level_count['user_age_level_buy_count'] / user_age_level_count['user_age_level_count']
    user_age_level_count['user_age_level'] = [index - 1000 for index in user_age_level_count['user_age_level']]

    feature = feature.merge(user_star_level_count, on='user_star_level', how='left').fillna(0)
    feature = feature.merge(user_occupation_id_count, on='user_occupation_id', how='left').fillna(0)
    feature = feature.merge(user_age_level_count, on='user_age_level', how='left').fillna(0)
    feature.iloc[:, -9:] = feature.iloc[:, -9:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    # print(feature)
    feature = feature.fillna(0)
    print(feature.iloc[:, 9:].shape)
    return feature.iloc[:, 9:]


'''特征提取验证'''
# extract_feature(train_data_one)


def drill_module(model='lightGBM', is_store=False, store_feature=False):
    '''训练模型'''
    if store_feature is True:
        '''获取数据集'''
        train_feature = extract_feature(train_data_one)
        train_feature = train_feature.append(extract_feature(train_data_two))
        train_feature = train_feature.append(extract_feature(train_data_three))
        train_feature = train_feature.append(extract_feature(train_data_four))
        # train_feature = train_feature.append(extract_feature(train_data_five))
        test_feature = extract_feature(test_data)
        validate_feature = extract_feature(validate_data)

        train_feature.to_csv('train_feature.csv', index=None)
        test_feature.to_csv('test_feature.csv', index=None)
        validate_feature.to_csv('validate_feature.csv', index=None)
    else:
        train_feature = pd.read_csv('train_feature.csv', low_memory=False)
        test_feature = pd.read_csv('test_feature.csv', low_memory=False)
        validate_feature = pd.read_csv('validate_feature.csv', low_memory=False)

    '''合并训练集测试集'''
    train_test_feature = train_feature.append(validate_feature)
    train_test_label = train_label.append(validate_label)

    '''归一化'''
    # train_feature = train_feature.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # # print(pd.DataFrame(train_feature))
    # test_feature = test_feature.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # validate_feature = validate_feature.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # train_test_feature = train_test_feature.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    ch2 = SelectKBest(chi2, k=90)   # 90:0.0803853705124  0.08128 || 95:0.080323993318  ||  80:0.0804241372249  0.08141  || 100: 0.0803716315055 || 85:0.0804166711167  ||  105:0.0803870235131  ||  70:0.0803751273261  ||  60:0.0804241372249#

    train_feature = ch2.fit_transform(train_feature, train_label)
    print(ch2.get_support(indices=True).tolist())
    # print(ch2.scores_)
    # print(ch2.pvalues_)
    # print('OK')
    # exit(0)
    test_feature = ch2.transform(test_feature)
    validate_feature = ch2.transform(validate_feature)
    train_test_feature = ch2.transform(train_test_feature)

    # print(pd.DataFrame(train_feature))
    # exit(0)

    if model == 'GBDT':
        '''GBDT module'''
        module = GradientBoostingClassifier(n_estimators=500, learning_rate=0.02, max_depth=4)   # 500 0.02 4 #

    elif model == 'xgboost':
        '''xgboost module'''
        # module = xgb.XGBClassifier(
        #         learning_rate=0.02,   # 0.05
        #         n_estimators=500,   # 500
        #         max_depth=4,   # 4
        #         gamma=0.1,
        #         subsample=0.7,
        #         objective='binary:logistic',  # reg:linear
        #         nthread=4,
        #     )
        # module.fit(train_feature, train_label)

        num_round = 500
        params = {
                  'booster': 'gbtree',
                  'max_depth': 4,
                  'colsample_bytree': 0.8,
                  'subsample': 0.8,
                  'eta': 0.03,
                  'silent': 1,
                  'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'min_child_weight': 0,
                  'scale_pos_weight': 1,
                  'lambda': 70
                }
        # 0.0803447896513
        dtrain = xgb.DMatrix(train_feature, label=train_label)
        test_feature = xgb.DMatrix(test_feature)
        validate_feature = xgb.DMatrix(validate_feature)
        module = xgb.train(params, dtrain, num_round)
        '''训练测试集模型'''
        d_train = xgb.DMatrix(train_test_feature, label=train_test_label)
        module_one = xgb.train(params, d_train, num_round)

    elif model == 'lightGBM':
        '''LightGBM module'''
        module = lgb.LGBMClassifier(num_leaves=8, max_depth=7, n_estimators=100)   # 0821982690783 8 7 #

    if model != 'xgboost':
        module.fit(train_feature, train_label)
        module_one.fit(train_test_feature, train_test_label)
        result = module_one.predict_proba(test_feature)[:, 1]
    else:
        result = module_one.predict(test_feature)    # xgboost #


    # joblib.dump(module, "model.m")    # 保存模型 #
    # module = joblib.load("model.m")   # 读取模型 #
    result = [index * 0.88 for index in result]
    result = pd.DataFrame(result)
    result.columns = ['predicted_score']
    sample = ijcai_test_data.loc[:, ['instance_id']]
    sample['predicted_score'] = [index for index in result['predicted_score']]

    # print(module.feature_importances_)    # 输出特征重要性 #

    '''验证集'''
    if model != 'xgboost':
        validate_label_predict = module.predict_proba(validate_feature)[:, 1]
    else:
        validate_label_predict = module.predict(validate_feature)      # xgboost #

    validate_label_predict = pd.DataFrame(validate_label_predict)
    score = log_loss(validate_label, validate_label_predict)
    print(model, '-log_loss_score:', score)
    # plot_importance(module)
    # plt.show()
    if is_store == True:
        '''文件格式转换'''
        sample.to_csv('result.csv', index=None)
        f = open("result.csv", 'r')             # 返回一个文件对象
        r = open("result.txt", 'w')
        line = f.readline()             # 调用文件的 readline()方法
        while line:
            line = line.replace(',', ' ')
            r.write(line)
            line = f.readline()
        f.close()
        r.close()
        print(sample)
        print('sum = ', sum(list(sample['predicted_score'])))
        print('The result have updated!')
    else:
        pass

def main():
    '''源程序'''
    # module = 'GBDT'
    module = 'xgboost'      # offline:  #
    # module = 'lightGBM'     # offline: #

    drill_module(model=module, is_store=True, store_feature=False)                # iss_store:是否保存当前结果 #

    if False:
        result = pd.read_csv('result.csv', low_memory=True)
        print(result[result['predicted_score'] > 0.1])
        print(result[result['predicted_score'] > 0.1].shape)

if __name__ == '__main__':
    main()

# 112 features: 0.080259130851   122 features:0.0803470439128
