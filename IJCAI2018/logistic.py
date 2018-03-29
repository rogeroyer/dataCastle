#coding=utf-8
'''
Author:Roger
date:2018-03-2
offline:0.0825458137126
online:
module:Logistic
'''
import pandas as pd
import numpy as np
import time
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest    # 特征选择 #
from sklearn.feature_selection import chi2

'''读取数据集'''
ijcai_train_data = pd.read_csv('../data_pre_deal/ijcai_train_data.csv', low_memory=False)    # , nrows=1000 #
ijcai_test_data = pd.read_csv('../data_pre_deal/ijcai_test_data.csv', low_memory=False)

ijcai_train_data = ijcai_train_data.drop_duplicates(['instance_id'])   # 数据集去重 #
ijcai_train_data['context_timestamp'] = [time.localtime(index) for index in ijcai_train_data['context_timestamp']]   # 训练集时间戳处理 #
ijcai_train_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_train_data['context_timestamp']]
ijcai_train_data['timestamp_hour_mouth'] = [time.strftime("%H", index) for index in ijcai_train_data['context_timestamp']]

ijcai_test_data['context_timestamp'] = [time.localtime(index) for index in ijcai_test_data['context_timestamp']]    # 测试集时间戳处理 #
ijcai_test_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_test_data['context_timestamp']]
ijcai_test_data['timestamp_hour_mouth'] = [time.strftime("%H", index) for index in ijcai_test_data['context_timestamp']]

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

'''训练集'''
train_data_one = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180918') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180919')]
train_data_two = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180919') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180920')]
train_data_three = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180920') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180921')]
train_data_four = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180921') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180922')]
train_data_five = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180922') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180923')]
'''验证集'''
validate_data = ijcai_train_data[(ijcai_train_data['timestamp_year_mouth_day'] >= '20180923') & (ijcai_train_data['timestamp_year_mouth_day'] <= '20180924')]
# '''测试集'''
test_data = ijcai_train_data[ijcai_train_data['timestamp_year_mouth_day'] == '20180924']                # 可以尝试去23-25日的作为测试集 #
test_data = test_data.iloc[:, :-1].append(ijcai_test_data)                                              # 提取消费特征时需要去掉.iloc[:, :-1] #

'''训练集标签'''
train_label = pd.DataFrame()
train_label = train_label.append(train_data_one[train_data_one['timestamp_year_mouth_day'] == '20180919'].loc[:, ['is_trade']])
train_label = train_label.append(train_data_two[train_data_two['timestamp_year_mouth_day'] == '20180920'].loc[:, ['is_trade']])
train_label = train_label.append(train_data_three[train_data_three['timestamp_year_mouth_day'] == '20180921'].loc[:, ['is_trade']])
train_label = train_label.append(train_data_four[train_data_four['timestamp_year_mouth_day'] == '20180922'].loc[:, ['is_trade']])
train_label = train_label.append(train_data_five[train_data_five['timestamp_year_mouth_day'] == '20180923'].loc[:, ['is_trade']])

'''验证集标签'''
validate_label = validate_data[validate_data['timestamp_year_mouth_day'] == '20180924'].loc[:, ['is_trade']]

'''user 性别 职业 one-hot编码'''
user_gender_id = list(ijcai_train_data.drop_duplicates(['user_gender_id'])['user_gender_id'])
ijcai_test_data['user_occupation_id'] = [-2 if index == -1 else index for index in ijcai_test_data['user_occupation_id']]
user_occupation_id = list(ijcai_train_data.drop_duplicates(['user_occupation_id'])['user_occupation_id'])

'''取Series最大最小值'''
def max_nin(data):
    return data.max(), data.min()
'''提取交叉组合特征'''
def extract_cross_feature(dataset, use_index, use_value, function, feature_name):
    feature_temp = pd.pivot_table(dataset, index=use_index, values=use_value, aggfunc=function)
    feature_temp[use_index] = feature_temp.index
    feature_temp.columns = [feature_name, use_index]
    return feature_temp
'''求众数'''
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
'''求方差'''
def cal_var(group):
    return np.var(group)
'''求标准差'''
def cal_std(group):
    return np.std(group)

'''提取特征'''
def extract_feature(dataset):
    # global train_label
    date = dataset['timestamp_year_mouth_day'].max()
    # train_label = train_label.append(dataset_one.loc[:, ['is_trade']])
    dataset['shop_star_level'] = [index - 4999 for index in dataset['shop_star_level']]
    dataset['context_page_id'] = [index - 4000 for index in dataset['context_page_id']]
    dataset['user_star_level'] = [index - 3000 for index in dataset['user_star_level']]
    dataset_one = dataset[dataset['timestamp_year_mouth_day'] == date]  # Label区间 #
    '''属性特征   Attention:使用dataset_one数据集  '''
    feature = dataset_one.loc[:, ['instance_id', 'item_price_level']]
    attribute = ['item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
                 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level']

    for index in attribute:
        feature[index] = dataset_one.loc[:, [index]]

    shop_star_level_max, shop_star_level_min = max_nin(feature['shop_star_level'])
    feature['shop_star_level'] = [(index - shop_star_level_min)/(shop_star_level_max - shop_star_level_min) for index in feature['shop_star_level']]

    feature['user_age_level'] = [index - 1000 if index > 0 else 3 for index in feature['user_age_level']]

    '''item系列'''
    dataset_one['item_category_list_str'] = [index.split(';') for index in dataset_one['item_category_list']]  # 按;切片 #
    dataset_one['item_category_list_len'] = [len(index) for index in dataset_one['item_category_list_str']]  # 切片后字符串长度 #
    dataset_one['item_category_list_two'] = [index[1] for index in dataset_one['item_category_list_str']]
    item_category_list_one_two = ['8277336076276184272', '5755694407684602296', '4879721024980945592', '2011981573061447208',
                                  '7258015885215914736', '509660095530134768', '8710739180200009128', '5799347067982556520',
                                  '2642175453151805566', '2436715285093487584', '3203673979138763595',  '22731265849056483',
                                  '1968056100269760729']
    for index in item_category_list_one_two:
        feature[index] = [1 if (index == string) else 0 for string in dataset_one['item_category_list_two']]

    feature['item_category_list_len'] = dataset_one.loc[:, ['item_category_list_len']]
    '''item_property_list'''
    feature['item_property_list_len'] = [index.count(';')+1 for index in dataset_one['item_property_list']]
    item_property_list_max, item_property_list_min = max_nin(feature['item_property_list_len'])
    feature['item_property_list_num'] = [(index - item_property_list_min) / (item_property_list_max - item_property_list_min) for index in feature['item_property_list_len']]
    # log_loss_score : 0.0823714066223  lightGBM max_depth=5, n_estimators=80
    '''context_page_id'''
    context_page_max, context_page_min = max_nin(dataset_one['context_page_id'])
    feature['context_page_id'] = dataset_one.loc[:, ['context_page_id']]
    feature['context_page_normalize'] = [(index - context_page_min) / (context_page_max - context_page_min) for index in dataset_one['context_page_id']]
    feature['context_page_min'] = [1 if index == 1 else 0 for index in dataset_one['context_page_id']]
    feature['context_page_max'] = [1 if index == 20 else 0 for index in dataset_one['context_page_id']]
    # log_loss_score : 0.082305015012  lightGBM max_depth=5, n_estimators=80
    '''predict_category_property'''
    dataset_one['predict_category_property_ad'] = [index.split(';') for index in dataset_one['predict_category_property']]
    feature['predict_category_property_ad_num'] = [len(index) for index in dataset_one['predict_category_property_ad']]
    predict_category_property_max, predict_category_property_min = max_nin(feature['predict_category_property_ad_num'])
    feature['dataset_one_ad_normalize'] = [(index - predict_category_property_min) / (predict_category_property_max - predict_category_property_min) for index in feature['predict_category_property_ad_num']]
    feature['predict_category_property_ad_num_min'] = [1 if index >= 10 else 0 for index in feature['predict_category_property_ad_num']]
    # log_loss_score : 0.0822782240689  lightGBM max_depth=5, n_estimators=80
    '''user 性别 职业 特征'''
    for index in user_gender_id:
        feature[index] = [1 if string == index else 0 for string in dataset_one['user_gender_id']]

    for index in user_occupation_id:
        feature[index] = [1 if string == index else 0 for string in dataset_one['user_occupation_id']]

    # log_loss_score : 0.0823714066223  0.082609756162

    '''暂时无用特征'''
    # dataset_one['predict_category_property_ad_num_one'] = [index.count('/') for index in dataset_one['predict_category_property']]
    # dataset_one['predict_category_property_ad_num_two'] = [index.count('-1') for index in dataset_one['predict_category_property']]
    # predict_category_property_ad_num_one_max, predict_category_property_ad_num_one_min = max_nin(dataset_one['predict_category_property_ad_num_one'])
    # feature['predict_category_property_ad_num_one_normalize'] = [(index - predict_category_property_ad_num_one_min)/(predict_category_property_ad_num_one_max - predict_category_property_ad_num_one_min) for index in dataset_one['predict_category_property_ad_num_one']]
    # predict_category_property_ad_num_two_max, predict_category_property_ad_num_two_min = max_nin(dataset_one['predict_category_property_ad_num_two'])
    # feature['predict_category_property_ad_num_two_normalize'] = [(index - predict_category_property_ad_num_two_min)/(predict_category_property_ad_num_two_max - predict_category_property_ad_num_two_min) for index in dataset_one['predict_category_property_ad_num_two']]
    # feature['predict_category_property_rate'] = dataset_one['predict_category_property_ad_num_two'] / feature['predict_category_property_ad_num']
    # # 0.0823296840725
    # feature['predict_category_property_ad_num_two_min'] = [1 if index >= 9 else 0 for index in dataset_one['predict_category_property_ad_num_two']]
    # feature['predict_category_property_ad_num_one_min'] = [1 if index >= 9 else 0 for index in dataset_one['predict_category_property_ad_num_one']]
    # feature['predict_category_property_ad_num_one_max'] = [1 if index == 0 else 0 for index in dataset_one['predict_category_property_ad_num_one']]

    '''交叉组合统计特征'''
    statistic_feature = dataset_one.loc[:, ['user_id', 'shop_id', 'item_id', 'context_id', 'instance_id']]
    dataset['item_property_list_set'] = [index.count(';') + 1 for index in dataset['item_property_list']]
    '''user系列'''
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'user_star_level', 'count', 'user_user_star_level'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_brand_id', 'count', 'user_item_brand_id'), on='user_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_price_level', 'count', 'user_item_price_level1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_price_level', 'mean', 'user_item_price_level2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_price_level', call_mode, 'user_item_price_level3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_price_level', cal_var, 'user_item_price_level4'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_price_level', cal_std, 'user_item_price_level5'), on='user_id', how='left')

    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_property_list_set', 'count', 'user_item_property_list_set'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_property_list_set', 'mean', 'user_item_property_list_set1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_property_list_set', call_mode, 'user_item_property_list_set2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_property_list_set', cal_var, 'user_item_property_list_set3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_property_list_set', cal_std, 'user_item_property_list_set4'), on='user_id', how='left')

    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_sales_level', 'count', 'user_item_sales_level'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_sales_level', 'mean', 'user_item_sales_level1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_sales_level', call_mode, 'user_item_sales_level2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_sales_level', cal_var, 'user_item_sales_level3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_sales_level', cal_std, 'user_item_sales_level4'), on='user_id', how='left')

    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_collected_level', 'count', 'user_item_collected_level'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_collected_level', 'mean', 'user_item_collected_level1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_collected_level', call_mode, 'user_item_collected_level2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_collected_level', cal_var, 'user_item_collected_level3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_collected_level', cal_std, 'user_item_collected_level4'), on='user_id', how='left')

    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_pv_level', 'count', 'user_item_pv_level'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_pv_level', 'mean', 'user_item_pv_level1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_pv_level', call_mode, 'user_item_pv_level2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_pv_level', cal_var, 'user_item_pv_level3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'item_pv_level', cal_std, 'user_item_pv_level4'), on='user_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'context_page_id', 'count', 'user_context_page_id'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'context_page_id', 'mean', 'user_context_page_id1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'context_page_id', call_mode, 'user_context_page_id2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'context_page_id', cal_var, 'user_context_page_id3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'context_page_id', cal_std, 'user_context_page_id4'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'context_page_id', max, 'user_context_page_id5'), on='user_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_num_level', 'count', 'user_shop_review_num_level'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_num_level', 'mean', 'user_shop_review_num_level1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_num_level', call_mode, 'user_shop_review_num_level2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_num_level', cal_var, 'user_shop_review_num_level3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_num_level', cal_std, 'user_shop_review_num_level4'), on='user_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_positive_rate', 'mean', 'user_shop_review_positive_rate1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_positive_rate', call_mode, 'user_shop_review_positive_rate2'), on='user_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_positive_rate', cal_var, 'user_shop_review_positive_rate3'), on='user_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_review_positive_rate', cal_std, 'user_shop_review_positive_rate4'), on='user_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_star_level', 'mean', 'user_shop_star_level1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_star_level', call_mode, 'user_shop_star_level2'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_star_level', cal_var, 'user_shop_star_level3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_star_level', cal_std, 'user_shop_star_level4'), on='user_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_service', 'mean', 'user_shop_score_service1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_service', call_mode, 'user_shop_score_service2'), on='user_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_service', cal_var, 'user_shop_score_service3'), on='user_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_service', cal_std, 'user_shop_score_service4'), on='user_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_delivery', 'mean', 'user_shop_score_delivery1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_delivery', call_mode, 'user_shop_score_delivery2'), on='user_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_delivery', cal_var, 'user_shop_score_delivery3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_delivery', cal_std, 'user_shop_score_delivery4'), on='user_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_description', 'mean', 'user_shop_score_description1'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_description', call_mode, 'user_shop_score_description2'), on='user_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_description', cal_var, 'user_shop_score_description3'), on='user_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'user_id', 'shop_score_description', cal_std, 'user_shop_score_description4'), on='user_id', how='left')
    # log_loss_score : 0.0818511341436  0.08190

    '''shop系列'''
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'shop_review_num_level', 'count', 'shop_shop_review_num_level'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_property_list_set', 'count', 'shop_item_property_list_set'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_brand_id', 'count', 'shop_item_brand_id'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_property_list_set', 'mean', 'shop_item_property_list_set1'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_property_list_set', call_mode, 'shop_item_property_list_set2'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_property_list_set', cal_var, 'shop_item_property_list_set3'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_property_list_set', cal_std, 'shop_item_property_list_set4'), on='shop_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_price_level', 'mean', 'shop_item_price_level1'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_price_level', call_mode, 'shop_item_price_level2'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_price_level', cal_var, 'shop_item_price_level3'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_price_level', cal_std, 'shop_item_price_level4'), on='shop_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_sales_level', 'mean', 'shop_item_sales_level1'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_sales_level', call_mode, 'shop_item_sales_level2'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_sales_level', cal_var, 'shop_item_sales_level3'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_sales_level', cal_std, 'shop_item_sales_level4'), on='shop_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_collected_level', 'mean', 'shop_item_collected_level1'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_collected_level', call_mode, 'shop_item_collected_level2'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_collected_level', cal_var, 'shop_item_collected_level3'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_collected_level', cal_std, 'shop_item_collected_level4'), on='shop_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_pv_level', 'mean', 'shop_item_pv_level1'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_pv_level', call_mode, 'shop_item_pv_level2'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_pv_level', cal_var, 'shop_item_pv_level3'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'item_pv_level', cal_std, 'shop_item_pv_level4'), on='shop_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'user_star_level', 'count', 'shop_user_star_level'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'user_star_level', 'mean', 'shop_user_star_level1'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'user_star_level', call_mode, 'shop_user_star_level2'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'user_star_level', cal_var, 'shop_user_star_level3'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'user_star_level', cal_std, 'shop_user_star_level4'), on='shop_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'context_page_id', 'count', 'shop_context_page_id'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'context_page_id', 'mean', 'shop_context_page_id1'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'context_page_id', call_mode, 'shop_context_page_id2'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'context_page_id', cal_var, 'shop_context_page_id3'), on='shop_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'shop_id', 'context_page_id', cal_std, 'shop_context_page_id4'), on='shop_id', how='left')

    '''item系列'''
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'item_property_list', 'count', 'item_item_property_list'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'user_star_level', 'count', 'item_user_star_level'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'user_star_level', 'mean', 'item_user_star_level1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'user_star_level', call_mode, 'item_user_star_level2'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'user_star_level', cal_var, 'item_user_star_level3'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'user_star_level', cal_std, 'item_user_star_level4'), on='item_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'context_page_id', 'count', 'item_context_page_id'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'context_page_id', 'mean', 'item_context_page_id1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'context_page_id', call_mode, 'item_context_page_id2'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'context_page_id', cal_var, 'item_context_page_id3'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'context_page_id', cal_std, 'item_context_page_id4'), on='item_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_num_level', 'count', 'item_shop_review_num_level'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_num_level', 'mean', 'item_shop_review_num_level1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_num_level', call_mode, 'item_shop_review_num_level2'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_num_level', cal_var, 'item_shop_review_num_level3'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_num_level', cal_std, 'item_shop_review_num_level4'), on='item_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_positive_rate', 'mean', 'item_shop_review_positive_rate1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_positive_rate', call_mode, 'item_shop_review_positive_rate2'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_positive_rate', cal_var, 'item_shop_review_positive_rate3'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_review_positive_rate', cal_std, 'item_shop_review_positive_rate4'), on='item_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_star_level', 'mean', 'item_shop_star_level1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_star_level', call_mode, 'item_shop_star_level2'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_star_level', cal_var, 'item_shop_star_level3'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_star_level', cal_std, 'item_shop_star_level4'), on='item_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_service', 'mean', 'item_shop_score_service1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_service', call_mode, 'item_shop_score_service2'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_service', cal_var, 'item_shop_score_service3'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_service', cal_std, 'item_shop_score_service4'), on='item_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_delivery', 'mean', 'item_shop_score_delivery1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_delivery', call_mode, 'item_shop_score_delivery2'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_delivery', cal_var, 'item_shop_score_delivery3'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_delivery', cal_std, 'item_shop_score_delivery4'), on='item_id', how='left')

    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_description', 'mean', 'item_shop_score_description1'), on='item_id', how='left')
    statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_description', call_mode, 'item_shop_score_description2'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_description', cal_var, 'item_shop_score_description3'), on='item_id', how='left')
    # statistic_feature = statistic_feature.merge(extract_cross_feature(dataset, 'item_id', 'shop_score_description', cal_std, 'item_shop_score_description4'), on='item_id', how='left')

    statistic_feature = statistic_feature.iloc[:, 4:]
    feature = feature.merge(statistic_feature, on='instance_id', how='left')

    print(feature.shape)
    return feature.iloc[:, 1:]

'''训练模型'''
def drill_module(model = 'lightGBM', is_store=False, get_new_feature=False, features=100):
    '''获取数据集'''
    if get_new_feature == True:
        train_feature = extract_feature(train_data_one)
        train_feature = train_feature.append(extract_feature(train_data_two))
        train_feature = train_feature.append(extract_feature(train_data_three))
        # train_feature = extract_feature(train_data_three)
        train_feature = train_feature.append(extract_feature(train_data_four))
        train_feature = train_feature.append(extract_feature(train_data_five))
        test_feature = extract_feature(test_data)
        validate_feature = extract_feature(validate_data)

        train_feature.to_csv('train_feature.csv', index=None)
        test_feature.to_csv('test_feature.csv', index=None)
        validate_feature.to_csv('validate_feature.csv', index=None)
    else:
        train_feature = pd.read_csv('train_feature.csv', low_memory=False)
        test_feature = pd.read_csv('test_feature.csv', low_memory=False)
        validate_feature = pd.read_csv('validate_feature.csv', low_memory=False)
        '''归一化'''
        def normalize(dataset):
            for num in dataset.columns:
                max_num = dataset[num].max()
                min_num = dataset[num].min()
                dataset[num] = [(index - min_num) / (max_num - min_num) for index in dataset[num]]
            return dataset

        # train_feature = normalize(train_feature)
        # test_feature = normalize(test_feature)
        # validate_feature = normalize(validate_feature)

        ch2 = SelectKBest(chi2, k=features)
        train_feature = ch2.fit_transform(train_feature, train_label)
        test_feature = ch2.transform(test_feature)
        validate_feature = ch2.transform(validate_feature)
        print(train_feature.shape)
        print(test_feature.shape)
        print(validate_feature.shape)
        '''热力图'''
        # data = pd.DataFrame(train_feature).corr()
        # sns.heatmap(data)
        # plt.show()

    if model == 'GBDT':
        '''GBDT module'''
        module = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5)

    elif model == 'xgboost':
        '''xgboost module'''
        module = xgb.XGBClassifier(
                learning_rate=0.07,
                n_estimators=500,
                max_depth=3,
                gamma=0.1,
                subsample=0.7,
                objective='binary:logistic',
                nthread=4,
            )

    elif model == 'lightGBM':
        '''LightGBM module'''
        module = lgb.LGBMClassifier(
            num_leaves=63,          # num_leaves = 2^max_depth * 0.6 #
            max_depth=7,
            n_estimators=80,
            min_data_in_leaf=100,
            learning_rate=0.1,
            num_iterations=200
        )

    elif model == 'tree':
        module = tree.DecisionTreeClassifier(criterion='gini', max_depth=2)

    elif model == 'forest':
        module = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini')

    elif model == 'Logistic':
        module = LogisticRegression(penalty='l2', solver='sag', max_iter=500, random_state=42, n_jobs=4)
        # module = linear_model.SGDClassifier(learning_rate='optimal')

    elif model == 'SVM':
        module = SVC(C=1.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, tol=0.001)

    module.fit(train_feature, train_label)
    result = module.predict_proba(test_feature)[:, 1]
    # joblib.dump(module, "model.m")    # 保存模型 #
    # module = joblib.load("model.m")   # 读取模型 #
    result = pd.DataFrame(result)
    result.columns = ['predicted_score']
    sample = ijcai_test_data.loc[:, ['instance_id']]
    sample['predicted_score'] = [index for index in result['predicted_score']]
    # print(sample)

    '''验证集'''
    validate_label_predict = module.predict_proba(validate_feature)[:, 1]
    validate_label_predict = pd.DataFrame(validate_label_predict)
    score = log_loss(validate_label, validate_label_predict)

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
        print('The result have updated!')

    print(sample[sample['predicted_score'] > 0.1])
    print(sample[sample['predicted_score'] > 0.1].shape)
    print(model, '-log_loss_score:', score)

def main():
    '''源程序'''
    # module = 'GBDT'         # offline: 0.0826341432788 #
    # module = 'xgboost'      # offline: 0.0827897548881 #
    # module = 'lightGBM'     # offline: 0.0824507211198   features=120 #
    # module = 'tree'         # offline: 0.0843034628994 #
    # module = 'forest'       # offline: 0.0833564894226 #
    module = 'Logistic'
    # module = 'SVM'
    drill_module(model=module, is_store=True, get_new_feature=False, features=150)                # is_store:是否保存当前结果 True False   get_train_feature是否保存当前训练特征    #

if __name__ == '__main__':
    main()

# sample = pd.read_csv('result.csv', low_memory=False)
# sample['predicted_score'] = [index + 0.2 if index > 0.1 else index for index in sample['predicted_score']]
# print(sample[sample['predicted_score'] > 0.1])
# print(sample[sample['predicted_score'] > 0.1].shape)

# train_feature = pd.read_csv('train_feature.csv', low_memory=False)
# print(train_feature.columns)
# # print(np.isnan(train_feature).any())
