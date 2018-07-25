# -*- coding:utf-8 -*-

import gc
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

path = 'dataSet//'

ccf_off_train_data = pd.read_csv(path + 'ccf_off_train_data.csv', encoding='utf-8', dtype={0: np.str, 1: np.str, 2: np.str, 3: np.str})
ccf_off_test_data = pd.read_csv(path + 'ccf_off_test_data.csv', encoding='utf-8', dtype={0: np.str, 1: np.str, 2: np.str, 3: np.str})

# 数据集去重 #
ccf_off_train_data = ccf_off_train_data.drop_duplicates()
# ccf_off_test_data = ccf_off_test_data.drop_duplicates()   # 有重复值 #


def return_set_len(group):
    return len(set(group))


def return_list(group):
    return list(group)


def return_set(group):
    return set(group)


def calc_auc_score(data_frame, predict):
    '''
    :param data_frame:['Coupon_id', 'label']
    :param predict: predict = []
    :return:auc score
    '''
    def return_score(label, predict):
        return roc_auc_score(label, predict)

    data_frame['predict'] = [index for index in predict]
    data_frame_label = pd.pivot_table(data_frame, index='Coupon_id', values=['label'], aggfunc=return_list).reset_index()
    data_frame_predict = pd.pivot_table(data_frame, index='Coupon_id', values=['predict'], aggfunc=return_list).reset_index()
    data_frame_label = data_frame_label.merge(data_frame_predict, on='Coupon_id', how='left')
    data_frame_label['score'] = data_frame_label.apply(lambda index: return_score(index.label, index.predict),axis=1)
    # print(data_frame_label)
    return np.mean(data_frame_label['score'])


def extract_label_section_features(start_date, end_date, ccf_off_data):
    features = ccf_off_data[(ccf_off_data['Date_received'] >= start_date) & (ccf_off_data['Date_received'] < end_date)]
    features['distance_inverse'] = 10 - features['Distance']

    def calc_discount_type(words):
        if words is np.nan:
            return 0, np.nan, np.nan, np.nan
        if words.find('.') >= 0:
            return 0, -1, -1, float(words)
        else:
            up = int(words.split(':')[1])
            down = int(words.split(':')[0])
            return 1, down, up, up / down

    features['discount_type'] = features['Discount_rate'].map(lambda index: calc_discount_type(index))
    features['discount_type_man'] = features['discount_type'].map(lambda index: index[0])
    features['discount_type_down'] = features['discount_type'].map(lambda index: index[1])
    features['discount_type_up'] = features['discount_type'].map(lambda index: index[2])
    features['discount_type_rate'] = features['discount_type'].map(lambda index: index[3])
    features = features.drop(['discount_type'], axis=1)

    # 距离和折扣率的排序特征 #
    features['user_distance_rank'] = features.groupby('User_id')['Distance'].rank(ascending=False, method='dense')
    features['user_distance_rank_reverse'] = features.groupby('User_id')['Distance'].rank(ascending=True, method='dense')
    features['user_discount_rate_rank'] = features.groupby('User_id')['discount_type_rate'].rank(ascending=False, method='dense')
    features['user_discount_rate_rank_reverse'] = features.groupby('User_id')['discount_type_rate'].rank(ascending=True, method='dense')
    features['merchant_distance_rank'] = features.groupby('Merchant_id')['Distance'].rank(ascending=False, method='dense')
    features['merchant_distance_rank_reverse'] = features.groupby('Merchant_id')['Distance'].rank(ascending=True, method='dense')
    features['merchant_discount_rate_rank'] = features.groupby('Merchant_id')['discount_type_rate'].rank(ascending=False, method='dense')
    features['merchant_discount_rate_rank_reverse'] = features.groupby('Merchant_id')['discount_type_rate'].rank(ascending=True, method='dense')
    features['coupon_distance_rank'] = features.groupby('Coupon_id')['Distance'].rank(ascending=False, method='dense')
    features['coupon_distance_rank_reverse'] = features.groupby('Coupon_id')['Distance'].rank(ascending=True, method='dense')
    # print(features)
    return features.drop(['Discount_rate', 'Date_received'], axis=1)


'''标签区间'''
# extract_label_section_features('2016-04-01', '2016-05-01', ccf_off_train_data)
# extract_label_section_features('2016-05-15', '2016-06-15', ccf_off_train_data)
# extract_label_section_features('2016-07-01', '2016-07-32', ccf_off_test_data)


def extract_feature_section_features(start_date, end_date, ccf_off_data, features):
    ccf_data = ccf_off_data[(ccf_off_data['Date_received'] >= start_date) & (ccf_off_data['Date_received'] < end_date)]
    coupon_isnull_data = ccf_off_data[ccf_off_data['Date_received'].isnull()]      # 未领券的消费记录 #
    ccf_data_two = coupon_isnull_data[(coupon_isnull_data['Date'] >= start_date) & (coupon_isnull_data['Date'] < end_date)]       # 该区间内未领券但消费了的用户 #
    del coupon_isnull_data
    gc.collect()

    ######################## 领券记录 #######################
    # 用户基础统计特征 #
    user_merchant_cnt = pd.pivot_table(ccf_data, index='User_id', values='Merchant_id', aggfunc='count').reset_index().rename(columns={'Merchant_id': 'user_merchant_cnt'})
    features = features.merge(user_merchant_cnt, on='User_id', how='left')
    user_merchant_set_len = pd.pivot_table(ccf_data, index='User_id', values='Merchant_id', aggfunc=return_set_len).reset_index().rename(columns={'Merchant_id': 'user_merchant_set_len'})
    user_coupon_set_len = pd.pivot_table(ccf_data, index='User_id', values='Coupon_id', aggfunc=return_set_len).reset_index().rename(columns={'Coupon_id': 'user_coupon_set_len'})
    user_discount_rate_set_len = pd.pivot_table(ccf_data, index='User_id', values='Discount_rate', aggfunc=return_set_len).reset_index().rename(columns={'Discount_rate': 'user_discount_rate_set_len'})
    user_discount_type_rate = pd.pivot_table(features, index='User_id', values='discount_type_rate', aggfunc=[np.mean, np.max, np.min, np.median, np.var]).reset_index()
    user_discount_type_rate.columns = ['User_id', 'discount_type_rate_mean', 'discount_type_rate_max', 'discount_type_rate_min', 'discount_type_rate_median', 'discount_type_rate_var']
    user_distance = pd.pivot_table(features, index='User_id', values='Distance', aggfunc=[np.mean, np.max, np.min, np.median, np.var]).reset_index()
    user_distance.columns = ['User_id', 'user_distance_mean', 'user_distance_max', 'user_distance_min', 'user_distance_median', 'user_distance_var']
    features = features.merge(user_merchant_set_len, on='User_id', how='left')
    features = features.merge(user_coupon_set_len, on='User_id', how='left')
    features = features.merge(user_discount_rate_set_len, on='User_id', how='left')
    features = features.merge(user_discount_type_rate, on='User_id', how='left')
    features = features.merge(user_distance, on='User_id', how='left')

    # 商铺基础统计特征 #
    merchant_cnt = pd.pivot_table(ccf_data, index='Merchant_id', values='User_id', aggfunc='count').reset_index().rename(columns={'User_id': 'merchant_cnt'})
    features = features.merge(merchant_cnt, on='Merchant_id', how='left')
    merchant_discount_rate = pd.pivot_table(features, index='Merchant_id', values='discount_type_rate', aggfunc=[np.mean, np.max, np.min, np.median, np.var]).reset_index()
    merchant_discount_rate.columns = ['Merchant_id', 'merchant_discount_rate_mean', 'merchant_discount_rate_max', 'merchant_discount_rate_min', 'merchant_discount_rate_median', 'merchant_discount_rate_var']
    merchant_distance = pd.pivot_table(features, index='Merchant_id', values='Distance', aggfunc=[np.mean, np.max, np.min, np.median, np.var]).reset_index()
    merchant_distance.columns = ['Merchant_id', 'merchant_distance_mean', 'merchant_distance_max', 'merchant_distance_min', 'merchant_distance_median', 'merchant_distance_var']
    features = features.merge(merchant_discount_rate, on='Merchant_id', how='left')
    features = features.merge(merchant_distance, on='Merchant_id', how='left')

    # 优惠券基础统计特征 #
    coupon_cnt = pd.pivot_table(ccf_data, index='Coupon_id', values='User_id', aggfunc='count').reset_index().rename(columns={'User_id': 'coupon_cnt'})
    features = features.merge(coupon_cnt, on='Coupon_id', how='left')
    coupon_distance = pd.pivot_table(features, index='Coupon_id', values='Distance', aggfunc=[np.mean, np.max, np.min, np.median, np.var]).reset_index()
    coupon_distance.columns = ['Coupon_id', 'coupon_distance_mean', 'coupon_distance_max', 'coupon_distance_min', 'coupon_distance_median', 'coupon_distance_var']
    features = features.merge(coupon_distance, on='Coupon_id', how='left')

    # 用户-商铺基础统计特征 #
    user_merchant_coupon_cnt = pd.pivot_table(ccf_data, index=['User_id', 'Merchant_id'], values='Coupon_id', aggfunc=return_set_len).reset_index().rename(columns={'Coupon_id': 'user_merchant_coupon_cnt'})
    user_merchant_coupon_discount_rate = pd.pivot_table(features, index=['User_id', 'Merchant_id'], values='discount_type_rate', aggfunc=[np.mean, np.max, np.min, np.median, np.var]).reset_index()
    user_merchant_coupon_discount_rate.columns = ['User_id', 'Merchant_id', 'user_merchant_coupon_discount_rate_mean', 'user_merchant_coupon_discount_rate_max', 'user_merchant_coupon_discount_rate_min', 'user_merchant_coupon_discount_rate_median', 'user_merchant_coupon_discount_rate_var']
    features = features.merge(user_merchant_coupon_cnt, on=['User_id', 'Merchant_id'], how='left')
    features = features.merge(user_merchant_coupon_discount_rate, on=['User_id', 'Merchant_id'], how='left')

    '''
    # 用户-优惠券基础统计特征 #
    user_coupon_merchant_cnt = pd.pivot_table(ccf_data, index=['User_id', 'Coupon_id'], values='Merchant_id', aggfunc=return_set_len).reset_index().rename(columns={'Merchant_id': 'user_coupon_merchant_cnt'})
    user_coupon_merchant_discount_rate = pd.pivot_table(features, index=['User_id', 'Coupon_id'], values='discount_type_rate', aggfunc=[np.mean, np.max, np.min, np.median, np.var]).reset_index()
    user_coupon_merchant_discount_rate.columns = ['User_id', 'Coupon_id', 'user_coupon_merchant_discount_rate_mean', 'user_coupon_merchant_discount_rate_max', 'user_coupon_merchant_discount_rate_min', 'user_coupon_merchant_discount_rate_median', 'user_coupon_merchant_discount_rate_var']
    features = features.merge(user_coupon_merchant_cnt, on=['User_id', 'Coupon_id'], how='left')
    features = features.merge(user_coupon_merchant_discount_rate, on=['User_id', 'Coupon_id'], how='left')
    '''

    ######################## 领券消费行为特征 ######################
    # 领取但未核销的张数   user_merchant_cnt为用户领券记录 #
    def nan_count(group):
        return list(group).count(np.nan)

    def label_count(group):
        return list(group).count(1)

    # 用户领券消费行为 #
    user_coupon_not_use = ccf_data.groupby(by=['User_id'], as_index=False)['Date'].agg({'user_coupon_not_use': nan_count})
    user_coupon_use_cnt = ccf_data.groupby(by=['User_id'], as_index=False)['label'].agg({'user_coupon_use_cnt': label_count})
    features = features.merge(user_coupon_not_use, on='User_id', how='left')   # 用户未使用券的数量 #
    features = features.merge(user_coupon_use_cnt, on='User_id', how='left')   # 用户在15天内使用券的数量 #
    features['user_coupon_use_behind'] = features['user_merchant_cnt'] - features['user_coupon_not_use'] - features['user_coupon_use_cnt']    # 超出15天使用券的数量 #
    features['user_coupon_not_use_rate'] = features['user_coupon_not_use'] / features['user_merchant_cnt']   # 比率特征 #
    features['user_coupon_use_cnt_rate'] = features['user_coupon_use_cnt'] / features['user_merchant_cnt']   # 转化率 #
    features['user_coupon_use_behind_rate'] = features['user_coupon_use_behind'] / features['user_merchant_cnt']
    user_coupon_label_max = ccf_data.groupby(by=['User_id'], as_index=False)['label'].agg({'user_coupon_label_max': np.max})   # 用户是否有15天消费的券 #
    user_coupon_all_use_in_15 = ccf_data.groupby(by=['User_id'], as_index=False)['label'].agg({'user_coupon_all_use_in_15': np.min})   # 所有券是否全在15天内消费使用 #
    features = features.merge(user_coupon_label_max, on='User_id', how='left')
    features = features.merge(user_coupon_all_use_in_15, on='User_id', how='left')

    # 商铺消费行为 #
    merchant_coupon_not_use = ccf_data.groupby(by=['Merchant_id'], as_index=False)['Date'].agg({'merchant_coupon_not_use': nan_count})
    merchant_coupon_use_cnt = ccf_data.groupby(by=['Merchant_id'], as_index=False)['label'].agg({'merchant_coupon_use_cnt': label_count})
    features = features.merge(merchant_coupon_not_use, on='Merchant_id', how='left')  # 商铺未使用券的数量 #
    features = features.merge(merchant_coupon_use_cnt, on='Merchant_id', how='left')  # 商铺在15天内被使用券的数量 #
    features['merchant_coupon_use_behind'] = features['merchant_cnt'] - features['merchant_coupon_not_use'] - features['merchant_coupon_use_cnt']    # 超出15天使用券的数量 #
    features['merchant_coupon_not_use_rate'] = features['merchant_coupon_not_use'] / features['merchant_cnt']   # 比率特征 #
    features['merchant_coupon_use_cnt_rate'] = features['merchant_coupon_use_cnt'] / features['merchant_cnt']   # 转化率 #
    features['merchant_coupon_use_behind_rate'] = features['merchant_coupon_use_behind'] / features['merchant_cnt']
    merchant_coupon_label_max = ccf_data.groupby(by=['Merchant_id'], as_index=False)['label'].agg({'merchant_coupon_label_max': np.max})   # 用户是否有15天消费的券 #
    merchant_coupon_all_use_in_15 = ccf_data.groupby(by=['Merchant_id'], as_index=False)['label'].agg({'merchant_coupon_all_use_in_15': np.min})   # 所有券是否全在15天内消费使用 #
    features = features.merge(merchant_coupon_label_max, on='Merchant_id', how='left')
    features = features.merge(merchant_coupon_all_use_in_15, on='Merchant_id', how='left')

    # 优惠券消费行为 #
    coupon_not_use = ccf_data.groupby(by=['Coupon_id'], as_index=False)['Date'].agg({'coupon_not_use': nan_count})
    coupon_use_cnt = ccf_data.groupby(by=['Coupon_id'], as_index=False)['label'].agg({'coupon_use_cnt': label_count})
    features = features.merge(coupon_not_use, on='Coupon_id', how='left')  # 用户未使用券的数量 #
    features = features.merge(coupon_use_cnt, on='Coupon_id', how='left')  # 用户在15天内使用券的数量 #
    features['coupon_use_behind'] = features['coupon_cnt'] - features['coupon_not_use'] - features['coupon_use_cnt']  # 超出15天使用券的数量 #
    features['coupon_not_use_rate'] = features['coupon_not_use'] / features['coupon_cnt']  # 比率特征 #
    features['coupon_use_cnt_rate'] = features['coupon_use_cnt'] / features['coupon_cnt']  # 转化率 #
    features['coupon_use_behind_rate'] = features['coupon_use_behind'] / features['coupon_cnt']
    coupon_label_max = ccf_data.groupby(by=['Coupon_id'], as_index=False)['label'].agg({'coupon_label_max': np.max})  # 用户是否有15天消费的券 #
    coupon_all_use_in_15 = ccf_data.groupby(by=['Coupon_id'], as_index=False)['label'].agg({'coupon_all_use_in_15': np.min})  # 所有券是否全在15天内消费使用 #
    features = features.merge(coupon_label_max, on='Coupon_id', how='left')
    features = features.merge(coupon_all_use_in_15, on='Coupon_id', how='left')

    # #######################  计算领券日期特征  ######################
    # # 周几领券 | 周几用券相关特征 #
    # ccf_data['week'] = [pd.to_datetime(index).weekday() + 1 for index in ccf_data['Date_received']]
    # user_week_cnt = pd.crosstab(ccf_data['User_id'], ccf_data['week']).reset_index()
    # user_week_cnt.columns = ['User_id', 'user_monday', 'user_tuesday', 'user_wednesday', 'user_thursday', 'user_friday', 'user_saturday', 'user_sunday']
    # features = features.merge(user_week_cnt, on='User_id', how='left')
    #
    # merchant_week_cnt = pd.crosstab(ccf_data['Merchant_id'], ccf_data['week']).reset_index()
    # merchant_week_cnt.columns = ['Merchant_id', 'merchant_monday', 'merchant_tuesday', 'merchant_wednesday', 'merchant_thursday', 'merchant_friday', 'merchant_saturday', 'merchant_sunday']
    # features = features.merge(merchant_week_cnt, on='Merchant_id', how='left')
    #
    # coupon_week_cnt = pd.crosstab(ccf_data['Coupon_id'], ccf_data['week']).reset_index()
    # coupon_week_cnt.columns = ['Coupon_id', 'coupon_monday', 'coupon_tuesday', 'coupon_wednesday', 'coupon_thursday', 'coupon_friday', 'coupon_saturday', 'coupon_sunday']
    # features = features.merge(coupon_week_cnt, on='Coupon_id', how='left')

    # print(features)
    # ccf_data = ccf_data[ccf_data['label'] == 0]
    # data = ccf_data['week'].value_counts().reset_index()
    # plt.bar(list(data['index']), list(data['week']), color='red')
    # plt.show()

    print(features.shape)
    return features


'''特征区间'''
# feature = extract_label_section_features('2016-04-01', '2016-05-01', ccf_off_train_data)
# extract_feature_section_features('2016-01-15', '2016-03-15', ccf_off_train_data, feature)

# extract_feature_section_features('2016-03-15', '2016-05-01', ccf_off_train_data)
# extract_feature_section_features('2016-05-01', '2016-06-15', ccf_off_train_data)

