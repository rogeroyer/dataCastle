#coding=utf-8
'''
Author:Roger
date:2018-04-08
offline:1
online:1
module:xgboost
'''

import pandas as pd
import numpy as np
import time
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


'''读取数据集'''
ijcai_train_data = pd.read_csv('../data_pre_deal/ijcai_train_data.csv', low_memory=False)    # , nrows=1000 #
ijcai_test_data = pd.read_csv('../data_pre_deal/ijcai_test_data.csv', low_memory=False)

ijcai_train_data = ijcai_train_data.drop_duplicates(['instance_id'])   # 数据集去重 #
ijcai_train_data['context_timestamp'] = [time.localtime(index) for index in ijcai_train_data['context_timestamp']]   # 训练集时间戳处理 #
ijcai_train_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_train_data['context_timestamp']]
ijcai_train_data['timestamp_hour_mouth'] = [time.strftime("%H", index) for index in ijcai_train_data['context_timestamp']]
ijcai_train_data['timestamp_hour_minute_second'] = [time.strftime("%H:%M:%S", index) for index in ijcai_train_data['context_timestamp']]

ijcai_test_data['context_timestamp'] = [time.localtime(index) for index in ijcai_test_data['context_timestamp']]    # 测试集时间戳处理 #
ijcai_test_data['timestamp_year_mouth_day'] = [time.strftime("%Y%m%d", index) for index in ijcai_test_data['context_timestamp']]
ijcai_test_data['timestamp_hour_mouth'] = [time.strftime("%H", index) for index in ijcai_test_data['context_timestamp']]
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

'''取Series最大最小值'''
def max_nin(data):
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
    dataset['shop_star_level'] = [index - 4999 for index in dataset['shop_star_level']]
    dataset['context_page_id'] = [index - 4000 for index in dataset['context_page_id']]
    dataset['user_star_level'] = [index - 3000 for index in dataset['user_star_level']]
    dataset_one = dataset[dataset['timestamp_year_mouth_day'] == date]       # Label区间 #
    dataset_two = dataset[dataset['timestamp_year_mouth_day'] != date]       # Feature区间 #
    # train_label = train_label.append(dataset_one.loc[:, ['is_trade']])


    '''属性特征   Attention:使用dataset_one数据集  '''
    feature = dataset_one.loc[:, ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'shop_id', 'context_id', 'item_price_level']]
    attribute = ['item_collected_level', 'item_pv_level', 'item_sales_level', 'shop_review_num_level', 'shop_review_positive_rate',
                 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'user_age_level', 'user_star_level']

    for index in attribute:
        feature[index] = dataset_one.loc[:, [index]]

    shop_star_level_max, shop_star_level_min = max_nin(feature['shop_star_level'])
    feature['shop_star_level'] = [(index - shop_star_level_min)/(shop_star_level_max - shop_star_level_min) for index in feature['shop_star_level']]
    feature['user_age_level'] = [index - 1000 if index > 0 else 3 for index in feature['user_age_level']]
    feature['user_star_level'] = [index - 3000 if index > 0 else 6 for index in feature['user_star_level']]

    '''item系列'''
    dataset_one['item_category_list_str'] = [index.split(';') for index in dataset_one['item_category_list']]  # 按;切片 #
    dataset_one['item_category_list_len'] = [len(index) for index in dataset_one['item_category_list_str']]  # 切片后字符串长度 #
    dataset_one['item_category_list_two'] = [index[1] for index in dataset_one['item_category_list_str']]
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

    '''排序特征'''

    def cal_conversion_rate(group):
        return list(group).count(1) / len(group)

    '''转化率特征'''
    item_conversion = pd.pivot_table(dataset_two, index='item_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_id', 'is_trade': 'item_trade'})
    item_brand_conversion = pd.pivot_table(dataset_two, index='item_brand_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_brand_id', 'is_trade': 'item_brand_trade'})
    item_city_conversion = pd.pivot_table(dataset_two, index='item_city_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_city_id', 'is_trade': 'item_city_trade'})

    shop_conversion = pd.pivot_table(dataset_two, index='shop_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'shop_id', 'is_trade': 'shop_trade'})
    user_conversion = pd.pivot_table(dataset_two, index='user_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'user_id', 'is_trade': 'user_trade'})

    feature = feature.merge(item_conversion, on='item_id', how='left').fillna(0)
    feature = feature.merge(item_brand_conversion, on='item_brand_id', how='left').fillna(0)
    feature = feature.merge(item_city_conversion, on='item_city_id', how='left').fillna(0)
    feature = feature.merge(shop_conversion, on='shop_id', how='left').fillna(0)
    feature = feature.merge(user_conversion, on='user_id', how='left').fillna(0)

    '''点击数'''
    user_click = extract_cross_feature(dataset_two, 'user_id', 'instance_id', 'count', 'user_click')
    shop_click = extract_cross_feature(dataset_two, 'shop_id', 'instance_id', 'count', 'shop_click')
    item_click = extract_cross_feature(dataset_two, 'item_id', 'instance_id', 'count', 'item_click')
    item_brand_click = extract_cross_feature(dataset_two, 'item_brand_id', 'instance_id', 'count', 'item_brand_click')
    item_city_click = extract_cross_feature(dataset_two, 'item_city_id', 'instance_id', 'count', 'item_city_click')

    feature = feature.merge(user_click, on='user_id', how='left').fillna(0)
    feature = feature.merge(shop_click, on='shop_id', how='left').fillna(0)
    feature = feature.merge(item_click, on='item_id', how='left').fillna(0)
    feature = feature.merge(item_brand_click, on='item_brand_id', how='left').fillna(0)
    feature = feature.merge(item_city_click, on='item_city_id', how='left').fillna(0)

    '''标准化 & 归一化'''
    # scaler = MinMaxScaler()
    # X = pd.DataFrame(scaler.fit_transform(statistic_feature.iloc[:, 7:]))
    # X.columns = list(statistic_feature.columns)[7:]
    # X['instance_id'] = statistic_feature['instance_id']
    # feature = feature.merge(X, on='instance_id', how='left')
    # print(feature)

    '''交叉组合统计特征'''
    dataset_two['item_property_list_set'] = [index.count(';') + 1 for index in dataset_two['item_property_list']]
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

    click_data['click_rank'] = click_data.groupby('user_id')['timestamp_hour_minute_second'].rank(method='min', ascending=True)
    feature['click_rank'] = [index for index in click_data['click_rank']]
    click_rank_max = pd.pivot_table(feature, index='user_id', values='click_rank', aggfunc=return_max_value).reset_index().rename(columns={'index': 'user_id', 'click_rank': 'click_max_value'})
    feature = feature.merge(click_rank_max, on='user_id', how='left')
    feature['is_min'] = feature[['click_rank']].apply(lambda x: 1 if x.click_rank == 1else 0, axis=1)
    feature['is_max'] = feature[['click_rank', 'click_max_value']].apply(lambda x: 1 if x.click_rank == x.click_max_value else 0, axis=1)

    feature = feature.fillna(0)
    print(feature.shape)
    return feature.iloc[:, 7:]


'''特征提取验证'''
# extract_feature(train_data_one)

'''训练模型'''
def drill_module(model = 'lightGBM', is_store=False, store_feature=False):
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

    if model == 'GBDT':
        '''GBDT module'''
        module = GradientBoostingClassifier(n_estimators=500, learning_rate=0.02, max_depth=4)   # 500 0.02 4 #

    elif model == 'xgboost':
        '''xgboost module'''
        module = xgb.XGBClassifier(
                learning_rate=0.02,   # 0.05
                n_estimators=500,   # 500
                max_depth=4,   # 4
                gamma=0.1,
                subsample=0.7,
                objective='binary:logistic',  # reg:linear
                nthread=4,
            )

    elif model == 'lightGBM':
        '''LightGBM module'''
        module = lgb.LGBMClassifier(num_leaves=8, max_depth=7, n_estimators=100)   # 0821982690783 8 7 #

    elif model == 'tree':
        module = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)

    elif model == 'forest':
        module = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini')

    elif model == 'Logistic':
        module = LogisticRegression(penalty='l2', solver='sag', max_iter=500, random_state=42, n_jobs=4)
        # module = linear_model.SGDClassifier(learning_rate='optimal')

    module.fit(train_feature, train_label)
    result = module.predict_proba(test_feature)[:, 1]
    # joblib.dump(module, "model.m")    # 保存模型 #
    # module = joblib.load("model.m")   # 读取模型 #
    result = pd.DataFrame(result)
    result.columns = ['predicted_score']
    sample = ijcai_test_data.loc[:, ['instance_id']]
    sample['predicted_score'] = [index for index in result['predicted_score']]
    # print(sample)

    # print(module.feature_importances_)    # 输出特征重要性 #

    '''验证集'''
    validate_label_predict = module.predict_proba(validate_feature)[:, 1]
    validate_label_predict = pd.DataFrame(validate_label_predict)
    score = log_loss(validate_label, validate_label_predict)
    print(model, '-log_loss_score:', score)

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
    else:
        pass

def main():
    '''源程序'''
    # module = 'GBDT'         # offline: 0.0826341432788 0.0814001613812 0.0807178761297#
    module = 'xgboost'      # offline: 0.082280216083 0822170063033 0.0806614780394 0.0805902313631#
    # module = 'lightGBM'     # offline: 0.0825307131425 0.0807135395986 0.0805965857494#
    # module = 'tree'         # offline: 0.0843034628994 #
    # module = 'forest'       # offline: 0.0833564894226 #
    # module = 'Logistic'
    drill_module(model=module, is_store=True, store_feature=False)                # iss_store:是否保存当前结果 #

    if True:
        result = pd.read_csv('result.csv', low_memory=True)
        print(result[result['predicted_score'] > 0.1])
        print(result[result['predicted_score'] > 0.1].shape)

if __name__ == '__main__':
    main()

