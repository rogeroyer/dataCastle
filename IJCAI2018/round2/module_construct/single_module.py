#coding=utf-8

import time
import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
# from feature_engine.label_interval import *

attributes = ['instance_id', 'user_gender_id', 'user_occupation_id', 'item_price_level', 'item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level', 'item_category_list', 'context_timestamp', 'item_id', 'user_id', 'shop_id', 'item_brand_id', 'item_city_id', 'context_page_id']
ijcai_test_data_b = pd.read_csv(r'../dataset/round2_ijcai_18_test_b_20180510.txt', sep=' ', engine='c', usecols=attributes)


def read_data(nrows=None):
    global attributes
    global ijcai_test_data_b
    # attributes = ['instance_id', 'user_gender_id', 'user_occupation_id', 'item_price_level', 'item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level', 'item_category_list', 'context_timestamp', 'item_id', 'user_id', 'shop_id', 'item_brand_id', 'item_city_id', 'context_page_id']
    '''get datasets'''
    ijcai_test_data = pd.read_csv(r'../dataset/ijcai_test_data.csv', engine='c', usecols=attributes)
    # ijcai_test_data_b = pd.read_csv(r'../dataset/round2_ijcai_18_test_b_20180510.txt', sep=' ', engine='c', usecols=attributes)
    ijcai_test_data = pd.concat([ijcai_test_data, ijcai_test_data_b])
    attributes.append('is_trade')
    ijcai_train_data = pd.read_csv(r'../dataset/round2_train.txt', engine='c', nrows=nrows, sep=' ', usecols=attributes)
    print('数据读取完毕。。。')

    '''show records time'''
    ijcai_train_data['context_timestamp'] = [time.localtime(index) for index in ijcai_train_data['context_timestamp']]  # 训练集时间戳处理 #
    ijcai_train_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_train_data['context_timestamp']]
    ijcai_train_data['timestamp_hour'] = [int(time.strftime("%H", index)) for index in ijcai_train_data['context_timestamp']]
    ijcai_train_data['timestamp_hour_minute_second'] = [time.strftime("%H:%M:%S", index) for index in ijcai_train_data['context_timestamp']]

    ijcai_test_data['context_timestamp'] = [time.localtime(index) for index in ijcai_test_data['context_timestamp']]  # 测试集时间戳处理 #
    ijcai_test_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_test_data['context_timestamp']]
    ijcai_test_data['timestamp_hour'] = [int(time.strftime("%H", index)) for index in ijcai_test_data['context_timestamp']]
    ijcai_test_data['timestamp_hour_minute_second'] = [time.strftime("%H:%M:%S", index) for index in ijcai_test_data['context_timestamp']]

    '''item_category_list deal'''
    ijcai_train_data.loc[:, 'item_category_list_str'] = [index.split(';') for index in ijcai_train_data['item_category_list']]  # 按;切片 #
    ijcai_train_data.loc[:, 'item_category_list_len'] = [len(index) for index in ijcai_train_data['item_category_list_str']]  # 切片后字符串长度 #
    ijcai_train_data.loc[:, 'item_category_list_two'] = [index[1] for index in ijcai_train_data['item_category_list_str']]

    ijcai_test_data.loc[:, 'item_category_list_str'] = [index.split(';') for index in ijcai_test_data['item_category_list']]  # 按;切片 #
    ijcai_test_data.loc[:, 'item_category_list_len'] = [len(index) for index in ijcai_test_data['item_category_list_str']]  # 切片后字符串长度 #
    ijcai_test_data.loc[:, 'item_category_list_two'] = [index[1] for index in ijcai_test_data['item_category_list_str']]

    '''pre_deal'''
    ijcai_train_data.loc[:, 'user_age_level'] = [index - 1000 if index != -1 else index for index in ijcai_train_data['user_age_level']]
    # ijcai_train_data.loc[:, 'user_occupation_id'] = [index - 2000 if index != -1 else index for index in ijcai_train_data['user_occupation_id']]
    ijcai_train_data.loc[:, 'user_star_level'] = [index - 3000 if index != -1 else index for index in ijcai_train_data['user_star_level']]
    ijcai_train_data.loc[:, 'shop_star_level'] = [index - 5000 if index != -1 else index for index in ijcai_train_data['shop_star_level']]
    ijcai_train_data.loc[:, 'context_page_id'] = [index - 4000 if index != -1 else index for index in ijcai_train_data['context_page_id']]

    ijcai_test_data.loc[:, 'user_age_level'] = [index - 1000 if index != -1 else index for index in ijcai_test_data['user_age_level']]
    # ijcai_test_data.loc[:, 'user_occupation_id'] = [index - 2000 if index != -1 else index for index in ijcai_test_data['user_occupation_id']]
    ijcai_test_data.loc[:, 'user_star_level'] = [index - 3000 if index != -1 else index for index in ijcai_test_data['user_star_level']]
    ijcai_test_data.loc[:, 'shop_star_level'] = [index - 5000 if index != -1 else index for index in ijcai_test_data['shop_star_level']]
    ijcai_test_data.loc[:, 'context_page_id'] = [index - 4000 if index != -1 else index for index in ijcai_test_data['context_page_id']]

    # print(ijcai_train_data)

    '''smooth window method'''
    train_data_one = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180831']
    print(train_data_one.shape)
    train_data_two = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180901']
    print(train_data_two.shape)
    train_data_three = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180902']
    print(train_data_three.shape)
    train_data_four = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180903']
    print(train_data_four.shape)
    train_data_five = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180904']
    print(train_data_five.shape)
    train_data_six = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180905']
    print(train_data_six.shape)
    train_data_seven = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180906']    # 11-10 #
    train_data_eight = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180907']    # 11-11 #
    print('数据集划分完毕。。。')
    '''train_data'''
    train_data = train_data_one
    train_data = train_data.append(train_data_one)
    train_data = train_data.append(train_data_two)
    train_data = train_data.append(train_data_three)
    train_data = train_data.append(train_data_four)
    train_data = train_data.append(train_data_five)
    train_data = train_data.append(train_data_six)
    # train_data = train_data.append(train_data_seven)      #    #
    train_data.index = range(len(train_data))
    '''test_data'''
    test_data = ijcai_test_data
    test_data = test_data.append(train_data_eight)    # .drop(['is_trade'] #
    test_data.index = range(len(test_data))
    return train_data, test_data


def extract_cross_feature(dataset, use_index, use_value, function, feature_name):
    '''提取交叉组合特征'''
    feature_temp = pd.pivot_table(dataset, index=use_index, values=use_value, aggfunc=function)
    feature_temp[use_index] = feature_temp.index
    feature_temp.columns = [feature_name, use_index]
    return feature_temp


def extract_label_feature(data_set, name):
    data_set_morning = data_set[data_set['timestamp_hour'] < 12]
    data_set_afternoon = data_set[data_set['timestamp_hour'] >= 12]

    feature = data_set_afternoon[['item_id', 'user_id', 'shop_id', 'item_brand_id', 'item_city_id', 'user_occupation_id', 'instance_id', 'item_price_level', 'item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
                                'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'context_page_id', 'user_star_level']]

    '''user gender & occupation'''
    user_gender_id, user_occupation_id = [0, 1, 2, -1], [2002, 2003, 2004, 2005]

    for gender in user_gender_id:
        feature[gender] = [1 if index == gender else 0 for index in data_set_afternoon['user_gender_id']]

    # for occupation in user_occupation_id:
    #     feature[occupation] = [1 if index == occupation else 0 for index in data_set_afternoon['user_occupation_id']]

    # '''item_category_list features one-hot'''
    # item_category_list = ['1909641874861640857', '453525480366550911', '7226013370341271704', '6693726201323251689', '8769426218101861255', '7314150500379498593', '6670526099037031245', '3434689896486063330', '6254910033820815248', '1852600517265062354', '8468007938333142982', '926205401302902289', '5685690139879409547', '7423553047267511438', '2211060154630359130', '3089254302947620489', '768579787521575291', '5066527928272238333', '2871729383671301763', '1367177154073382718', '1147074168968532252', '8009556227083201357', '4911723539855588624', '1920084168104334820', '8841625760168847319', '597424223456586363', '1916390345133212703', '8468370105908620354', '3613783563199627217', '3348197449185791127', '394915394741096735']
    #
    # for item_category in item_category_list:
    #     feature[item_category] = [1 if index == item_category else 0 for index in data_set_afternoon['item_category_list_two']]

    feature.loc[:, 'item_category_list_len'] = [index for index in data_set_afternoon['item_category_list_len']]

    def cal_conversion_rate(group):
        return list(group).count(1) / len(group)

    def cal_click_count(group):
        return list(group).count(1)

    '''conversion features'''
    item_conversion = pd.pivot_table(data_set_morning, index='item_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_id', 'is_trade': 'item_trade'})
    item_brand_conversion = pd.pivot_table(data_set_morning, index='item_brand_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_brand_id', 'is_trade': 'item_brand_trade'})
    item_city_conversion = pd.pivot_table(data_set_morning, index='item_city_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_city_id', 'is_trade': 'item_city_trade'})
    shop_conversion = pd.pivot_table(data_set_morning, index='shop_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'shop_id', 'is_trade': 'shop_trade'})
    user_conversion = pd.pivot_table(data_set_morning, index='user_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'user_id', 'is_trade': 'user_trade'})
    '''user_shop_item_itembrand_itemcity_conservision'''
    # user_shop_trade_count = pd.pivot_table(data_set_morning, index=['user_id', 'shop_id'], values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'is_trade': 'user_shop_trade'})
    # user_item_trade_count = pd.pivot_table(data_set_morning, index=['user_id', 'item_id'], values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'is_trade': 'user_item_trade'})
    # user_item_brand_trade_count = pd.pivot_table(data_set_morning, index=['user_id', 'item_brand_id'], values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'is_trade': 'user_item_brand_trade_count'})
    # user_item_city_trade_count = pd.pivot_table(data_set_morning, index=['user_id', 'item_city_id'], values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'is_trade': 'user_item_city_trade_count'})
    # feature = feature.merge(user_shop_trade_count, on=['user_id', 'shop_id'], how='left')
    # feature = feature.merge(user_item_trade_count, on=['user_id', 'item_id'], how='left')
    # feature = feature.merge(user_item_brand_trade_count, on=['user_id', 'item_brand_id'], how='left')
    # feature = feature.merge(user_item_city_trade_count, on=['user_id', 'item_city_id'], how='left')

    '''trade count'''
    item_conversion_click = pd.pivot_table(data_set_morning, index='item_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'item_id', 'is_trade': 'item_trade_click'})
    item_brand_conversion_click = pd.pivot_table(data_set_morning, index='item_brand_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'item_brand_id', 'is_trade': 'item_brand_trade_click'})
    item_city_conversion_click = pd.pivot_table(data_set_morning, index='item_city_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'item_city_id', 'is_trade': 'item_city_trade_click'})
    shop_conversion_click = pd.pivot_table(data_set_morning, index='shop_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'shop_id', 'is_trade': 'shop_trade_click'})
    user_conversion_click = pd.pivot_table(data_set_morning, index='user_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'index': 'user_id', 'is_trade': 'user_trade_click'})
    '''user_shop_item_itembrand_itemcity_trade_click'''
    # user_shop_trade_click = pd.pivot_table(data_set_morning, index=['user_id', 'shop_id'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_shop_trade_click'})
    # user_item_trade_click = pd.pivot_table(data_set_morning, index=['user_id', 'item_id'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_item_trade_click'})
    # user_item_brand_trade_click = pd.pivot_table(data_set_morning, index=['user_id', 'item_brand_id'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_item_brand_trade_click'})
    # user_item_city_trade_click = pd.pivot_table(data_set_morning, index=['user_id', 'item_city_id'], values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_item_city_trade_click'})
    # feature = feature.merge(user_shop_trade_click, on=['user_id', 'shop_id'], how='left')
    # feature = feature.merge(user_item_trade_click, on=['user_id', 'item_id'], how='left')
    # feature = feature.merge(user_item_brand_trade_click, on=['user_id', 'item_brand_id'], how='left')
    # feature = feature.merge(user_item_city_trade_click, on=['user_id', 'item_city_id'], how='left')

    '''num count features'''
    # ['item_id', 'user_id', 'shop_id', 'item_brand_id', 'item_city_id']
    item_id_count = pd.pivot_table(data_set_morning, index='item_id', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'item_id_count'})
    user_id_count = pd.pivot_table(data_set_morning, index='user_id', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_id_count'})
    shop_id_count = pd.pivot_table(data_set_morning, index='shop_id', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'shop_id_count'})
    item_brand_id_count = pd.pivot_table(data_set_morning, index='item_brand_id', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'item_brand_id_count'})
    item_city_id_count = pd.pivot_table(data_set_morning, index='item_city_id', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'item_city_id_count'})
    '''user_shop_item_itembrand_itemcity_click_total'''
    # user_shop_count = pd.pivot_table(data_set_morning, index=['user_id', 'shop_id'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_shop_count'})
    # user_item_count = pd.pivot_table(data_set_morning, index=['user_id', 'item_id'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_item_count'})
    # user_item_brand_count = pd.pivot_table(data_set_morning, index=['user_id', 'item_brand_id'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_item_brand_count'})
    # user_item_city_count = pd.pivot_table(data_set_morning, index=['user_id', 'item_city_id'], values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_item_city_count'})
    # feature = feature.merge(user_shop_count, on=['user_id', 'shop_id'], how='left')
    # feature = feature.merge(user_item_count, on=['user_id', 'item_id'], how='left')
    # feature = feature.merge(user_item_brand_count, on=['user_id', 'item_brand_id'], how='left')
    # feature = feature.merge(user_item_city_count, on=['user_id', 'item_city_id'], how='left')
    '''merge'''
    feature = feature.merge(item_conversion, on='item_id', how='left')
    feature = feature.merge(item_brand_conversion, on='item_brand_id', how='left')
    feature = feature.merge(item_city_conversion, on='item_city_id', how='left')
    feature = feature.merge(shop_conversion, on='shop_id', how='left')
    feature = feature.merge(user_conversion, on='user_id', how='left')

    feature = feature.merge(item_conversion_click, on='item_id', how='left')
    feature = feature.merge(item_brand_conversion_click, on='item_brand_id', how='left')
    feature = feature.merge(item_city_conversion_click, on='item_city_id', how='left')
    feature = feature.merge(shop_conversion_click, on='shop_id', how='left')
    feature = feature.merge(user_conversion_click, on='user_id', how='left')

    feature = feature.merge(item_id_count, on='item_id', how='left')
    feature = feature.merge(user_id_count, on='user_id', how='left')
    feature = feature.merge(shop_id_count, on='shop_id', how='left')
    feature = feature.merge(item_brand_id_count, on='item_brand_id', how='left')
    feature = feature.merge(item_city_id_count, on='item_city_id', how='left')

    '''rank features'''
    '''返回最大值'''
    def return_max_value(group):
        return max(group)
    click_data = data_set_afternoon.loc[:, ['user_id', 'timestamp_hour_minute_second']]
    click_data['click_rank'] = click_data.groupby('user_id')['timestamp_hour_minute_second'].rank(method='min', ascending=True)
    click_data['click_rank_reverse'] = click_data.groupby('user_id')['timestamp_hour_minute_second'].rank(method='min', ascending=False)
    feature['click_rank'] = [index for index in click_data['click_rank']]
    feature['click_rank_reverse'] = [index for index in click_data['click_rank_reverse']]
    click_rank_max = pd.pivot_table(feature, index='user_id', values='click_rank', aggfunc=return_max_value).reset_index().rename(columns={'index': 'user_id', 'click_rank': 'click_max_value'})
    feature = feature.merge(click_rank_max, on='user_id', how='left')
    feature['is_min'] = feature[['click_rank']].apply(lambda x: 1 if x.click_rank == 1 else 0, axis=1)
    feature['is_max'] = feature[['click_rank', 'click_max_value']].apply(lambda x: 1 if x.click_rank == x.click_max_value else 0, axis=1)

    '''['item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
    'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level']'''
    '''user系列'''
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'item_price_level', 'mean', 'user_item_price_level2'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'item_sales_level', 'mean', 'user_item_sales_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'item_collected_level', 'mean', 'user_item_collected_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'item_pv_level', 'mean', 'user_item_pv_level1'), on='user_id', how='left')

    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'context_page_id', 'mean', 'user_context_page_id1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'shop_review_num_level', 'mean', 'user_shop_review_num_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'shop_review_positive_rate', 'mean', 'user_shop_review_positive_rate1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'shop_star_level', 'mean', 'user_shop_star_level1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'shop_score_service', 'mean', 'user_shop_score_service1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'shop_score_delivery', 'mean', 'user_shop_score_delivery1'), on='user_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'user_id', 'shop_score_description', 'mean', 'user_shop_score_description1'), on='user_id', how='left')

    '''shop系列'''
    feature = feature.merge(extract_cross_feature(data_set, 'shop_id', 'item_price_level', 'mean', 'shop_item_price_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'shop_id', 'item_sales_level', 'mean', 'shop_item_sales_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'shop_id', 'item_collected_level', 'mean', 'shop_item_collected_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'shop_id', 'item_pv_level', 'mean', 'shop_item_pv_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'shop_id', 'user_star_level', 'mean', 'shop_user_star_level1'), on='shop_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'shop_id', 'context_page_id', 'mean', 'shop_context_page_id1'), on='shop_id', how='left')

    '''item系列'''
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'user_star_level', 'mean', 'item_user_star_level1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'context_page_id', 'mean', 'item_context_page_id1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'shop_review_num_level', 'mean', 'item_shop_review_num_level1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'shop_review_positive_rate', 'mean', 'item_shop_review_positive_rate1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'shop_star_level', 'mean', 'item_shop_star_level1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'shop_score_service', 'mean', 'item_shop_score_service1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'shop_score_delivery', 'mean', 'item_shop_score_delivery1'), on='item_id', how='left')
    feature = feature.merge(extract_cross_feature(data_set, 'item_id', 'shop_score_description', 'mean', 'item_shop_score_description1'), on='item_id', how='left')

    # ''' 用户属性的点击数、购买量、转化率 '''
    # user_star_level_count = pd.pivot_table(data_set_morning, index='user_star_level', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_star_level_count'})
    # user_star_level_buy_count = pd.pivot_table(data_set_morning, index='user_star_level', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_star_level_buy_count'})
    # user_star_level_count = user_star_level_count.merge(user_star_level_buy_count, on='user_star_level', how='left')
    # user_star_level_count['conversion_rate1'] = user_star_level_count['user_star_level_buy_count'] / user_star_level_count['user_star_level_count']
    #
    # user_occupation_id_count = pd.pivot_table(data_set_morning, index='user_occupation_id', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_occupation_id_count'})
    # user_occupation_buy_count = pd.pivot_table(data_set_morning, index='user_occupation_id', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_occupation_buy_count'})
    # user_occupation_id_count = user_occupation_id_count.merge(user_occupation_buy_count, on='user_occupation_id', how='left')
    # user_occupation_id_count['conversion_rate2'] = user_occupation_id_count['user_occupation_buy_count'] / user_occupation_id_count['user_occupation_id_count']
    #
    # user_age_level_count = pd.pivot_table(data_set_morning, index='user_age_level', values='instance_id', aggfunc='count').reset_index().rename(columns={'instance_id': 'user_age_level_count'})
    # user_age_level_buy_count = pd.pivot_table(data_set_morning, index='user_age_level', values='is_trade', aggfunc=cal_click_count).reset_index().rename(columns={'is_trade': 'user_age_level_buy_count'})
    # user_age_level_count = user_age_level_count.merge(user_age_level_buy_count, on='user_age_level', how='left')
    # user_age_level_count['conversion_rate3'] = user_age_level_count['user_age_level_buy_count'] / user_age_level_count['user_age_level_count']
    # # user_age_level_count['user_age_level'] = [index - 1000 for index in user_age_level_count['user_age_level']]
    #
    # feature = feature.merge(user_star_level_count, on='user_star_level', how='left')
    # feature = feature.merge(user_occupation_id_count, on='user_occupation_id', how='left')
    # feature = feature.merge(user_age_level_count, on='user_age_level', how='left')
    # # feature.iloc[:, -9:] = feature.iloc[:, -9:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    feature = feature.fillna(0)
    # print(feature)
    print('The shape of features:', feature.shape)

    if name == 'train':
        return feature.iloc[:, 6:], data_set_afternoon[['is_trade']]
    else:
        return feature.iloc[:, 6:]


def lgb_model(store_features=False, store_result=False):
    if store_features is True:
        train_data, test_data = read_data(nrows=None)

        train_feature, train_label = extract_label_feature(train_data, name='train')
        test_feature = extract_label_feature(test_data, name='test')

        # print(train_feature, test_feature)

        train_feature.to_csv(r'train_feature.csv', index=None)
        test_feature.to_csv(r'test_feature.csv', index=None)
        train_label.to_csv(r'train_label.csv', index=None)
        print('data store completed.')

    else:
        train_feature = pd.read_csv(r'train_feature.csv')
        test_feature = pd.read_csv(r'test_feature.csv')
        train_label = pd.read_csv(r'train_label.csv')

        print(train_feature.shape)
        print(test_feature.shape)
        print(train_label.shape)
        print('read data completed.')

    train_feature = train_feature.iloc[:, 1:]
    sample = test_feature[['instance_id']]
    test_feature = test_feature.iloc[:, 1:]

    labels = np.array(train_label['is_trade'], dtype=np.int8)
    d_train = lgb.Dataset(train_feature, label=labels)
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'train_metric': True,
        'subsample': 0.8,
        'learning_rate': 0.05,
        'num_leaves': 96,
        'num_threads': 4,
        'max_depth': 5,
        'colsample_bytree': 0.8,
        'lambda_l2': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
    }

    rounds = 500
    watchlist = [d_train]
    bst = lgb.train(params=params, train_set=d_train, num_boost_round=rounds, valid_sets=watchlist, verbose_eval=10)
    predict = pd.DataFrame(bst.predict(test_feature), columns=['predicted_score'])
    sample['predicted_score'] = [index for index in predict['predicted_score']]
    print(sample)

    final = ijcai_test_data_b[['instance_id']]
    final = final.merge(sample, on='instance_id', how='left')
    print('SUM = ', sum(final['predicted_score']))


    if store_result is True:
        '''文件格式转换'''
        final.to_csv(r'lgb_sample.csv', index=None)     # final modified #
        print('结果已保存。。。')

        time_string = time.strftime('_%m%d', time.localtime(time.time()))
        file_name = 'lgb_result' + time_string + '.txt'
        f = open("lgb_sample.csv", 'r')  # 返回一个文件对象
        r = open(file_name, 'w')
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = line.replace(',', ' ')
            r.write(line)
            line = f.readline()
        f.close()
        r.close()
        print('The result have updated!')


def xgb_model(store_features=False, store_result=False):
    if store_features is True:
        train_data, test_data = read_data(nrows=None)

        train_feature, train_label = extract_label_feature(train_data, name='train')
        test_feature = extract_label_feature(test_data, name='test')

        # print(train_feature, test_feature)

        train_feature.to_csv(r'train_feature.csv', index=None)
        test_feature.to_csv(r'test_feature.csv', index=None)
        train_label.to_csv(r'train_label.csv', index=None)
        print('data store completed.')

    else:
        train_feature = pd.read_csv(r'train_feature.csv')
        test_feature = pd.read_csv(r'test_feature.csv')
        train_label = pd.read_csv(r'train_label.csv')

        # '''feature test'''
        # features = ['1909641874861640857', '453525480366550911', '7226013370341271704',
        #             '6693726201323251689', '8769426218101861255', '7314150500379498593',
        #             '6670526099037031245', '3434689896486063330', '6254910033820815248',
        #             '1852600517265062354', '8468007938333142982', '926205401302902289',
        #             '5685690139879409547', '7423553047267511438', '2211060154630359130',
        #             '3089254302947620489', '768579787521575291', '5066527928272238333',
        #             '2871729383671301763', '1367177154073382718', '1147074168968532252',
        #             '8009556227083201357', '4911723539855588624', '1920084168104334820',
        #             '8841625760168847319', '597424223456586363', '1916390345133212703',
        #             '8468370105908620354', '3613783563199627217', '3348197449185791127',
        #             '394915394741096735', '0', '1', '2', '-1', '2002', '2003', '2004', '2005']
        # train_feature = train_feature.drop(features, axis=1)
        # test_feature = test_feature.drop(features, axis=1)

        print(train_feature.shape)
        print(test_feature.shape)
        print(train_label.shape)
        print('read data completed.')

    train_feature = train_feature.iloc[:, 1:]
    sample = test_feature[['instance_id']]
    test_feature = test_feature.iloc[:, 1:]

    num_round = 500
    params = {
        'booster': 'gbtree',
        'max_depth': 6,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'eta': 0.03,
        'silent': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'min_child_weight': 16,
        'scale_pos_weight': 1
    }
    print('starting to construct module.')
    dtrain = xgb.DMatrix(train_feature, label=train_label['is_trade'])
    test_feature = xgb.DMatrix(test_feature)
    module = xgb.train(params, dtrain, num_round)
    print('模型训练完毕。。。')
    result = module.predict(test_feature)
    result = pd.DataFrame(result)
    result.columns = ['predicted_score']

    sample['predicted_score'] = [index for index in result['predicted_score']]
    print('结果已保存。。。')
    print(sample)

    final = ijcai_test_data_b[['instance_id']]
    final = final.merge(sample, on='instance_id', how='left')
    print('SUM = ', sum(final['predicted_score']))


    if store_result is True:
        '''文件格式转换'''
        final.to_csv(r'xgb_sample.csv', index=None)     # final modified #
        time_string = time.strftime('_%m%d', time.localtime(time.time()))
        file_name = 'xgb_result' + time_string + '.txt'
        f = open("xgb_sample.csv", 'r')  # 返回一个文件对象
        r = open(file_name, 'w')
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = line.replace(',', ' ')
            r.write(line)
            line = f.readline()
        f.close()
        r.close()
        print('The result have updated!')


def main():
    '''xgboost module'''
    # xgb_model(store_features=False, store_result=True)

    '''lightgbm module'''
    lgb_model(store_features=True, store_result=True)
    # [200] training's binary_logloss: 0.0555233
    # SUM = 16905.498844450693


if __name__ == '__main__':
    start_time = time.clock()
    main()
    print('This program spend time:', time.clock() - start_time, ' s')
    print('This program spend time:', (time.clock() - start_time) / 60, ' min')


