#coding=utf-8
'''
Author:Roger
date:2018-03-26
offline:0.0822725453071
online:2
module:xgboost
'''

import pandas as pd
import numpy as np
import time
import os
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


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

ijcai_test_data['shop_score_service'] = [index if index != -1 else ijcai_test_data['shop_score_service'].mean() for index in ijcai_test_data['shop_score_service']]
ijcai_test_data['shop_score_delivery'] = [index if index != -1 else ijcai_test_data['shop_score_delivery'].mean() for index in ijcai_test_data['shop_score_delivery']]
ijcai_test_data['shop_score_description'] = [index if index != -1 else ijcai_test_data['shop_score_description'].mean() for index in ijcai_test_data['shop_score_description']]

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

'''取Series最大最小值'''
def max_nin(data):
    return data.max(), data.min()

'''提取特征'''
def extract_feature(dataset):
    # global train_label
    date = dataset['timestamp_year_mouth_day'].max()
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
                                  '2642175453151805566', '2436715285093487584', '3203673979138763595',  '22731265849056483',
                                  '1968056100269760729']
    for index in item_category_list_one_two:
        feature[index] = [1 if (index == string) else 0 for string in dataset_one['item_category_list_two']]
    # feature['item_category_list_len'] = [index for index in dataset_one['item_category_list_len']]         # 可去特征 #
    feature['item_category_list_len'] = dataset_one.loc[:, ['item_category_list_len']]
    '''item_property_list'''
    feature['item_property_list_len'] = [index.count(';')+1 for index in dataset_one['item_property_list']]
    item_property_list_max, item_property_list_min = max_nin(feature['item_property_list_len'])
    feature['item_property_list_num'] = [(index - item_property_list_min) / (item_property_list_max - item_property_list_min) for index in feature['item_property_list_len']]
    # log_loss_score : 0.0823714066223

    # '''user统计特征'''
    # user_count = pd.pivot_table(dataset, index='user_id', values='user_gender_id', aggfunc='count')
    # user_count['user_id'] = user_count.index
    # user_count.columns = ['user_count', 'user_id']
    # dataset_one = dataset_one.merge(user_count, on='user_id', how='left')
    # feature['user_count'] = dataset_one.loc[:, ['user_count']]
    # # log_loss_score : 0.0823714066223  0.0824960359566

    '''排序特征'''

    def cal_conversion_rate(group):
        return list(group).count(1) / len(group)

    '''转化率特征'''
    item_conversion = pd.pivot_table(dataset_two, index='item_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_id', 'is_trade': 'item_trade'})
    item_brand_conversion = pd.pivot_table(dataset_two, index='item_brand_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_brand_id', 'is_trade': 'item_brand_trade'})
    item_city_conversion = pd.pivot_table(dataset_two, index='item_city_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'item_city_id', 'is_trade': 'item_city_trade'})

    shop_conversion = pd.pivot_table(dataset_two, index='shop_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'shop_id', 'is_trade': 'shop_trade'})
    user_conversion = pd.pivot_table(dataset_two, index='user_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'user_id', 'is_trade': 'user_trade'})
    context_conversion = pd.pivot_table(dataset_two, index='context_id', values='is_trade', aggfunc=cal_conversion_rate).reset_index().rename(columns={'index': 'context_id', 'is_trade': 'context_trade'})

    feature = feature.merge(item_conversion, on='item_id', how='left').fillna(0)
    feature = feature.merge(item_brand_conversion, on='item_brand_id', how='left').fillna(0)
    feature = feature.merge(item_city_conversion, on='item_city_id', how='left').fillna(0)
    feature = feature.merge(shop_conversion, on='shop_id', how='left').fillna(0)
    feature = feature.merge(user_conversion, on='user_id', how='left').fillna(0)
    feature = feature.merge(context_conversion, on='context_id', how='left').fillna(0)

    # print(item_conversion[item_conversion['item_trade'] != 0])
    # print(shop_conversion)
    # print(user_conversion)
    # print(context_conversion[context_conversion['context_trade'] != 0])
    # print(item_brand_conversion)
    # print(item_city_conversion)

    print(feature.shape)
    return feature.iloc[:, 7:]


'''特征提取验证'''
# print(extract_feature(train_data_one))

'''训练模型'''
def drill_module(model = 'lightGBM', is_store=False):
    '''获取数据集'''
    train_feature = extract_feature(train_data_one)
    train_feature = train_feature.append(extract_feature(train_data_two))
    train_feature = train_feature.append(extract_feature(train_data_three))
    train_feature = train_feature.append(extract_feature(train_data_four))
    train_feature = train_feature.append(extract_feature(train_data_five))
    test_feature = extract_feature(test_data)
    validate_feature = extract_feature(validate_data)

    if model == 'GBDT':
        '''GBDT module'''
        module = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=2)

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
        module = lgb.LGBMClassifier(num_leaves=63, max_depth=9, n_estimators=80)

    elif model == 'tree':
        module = tree.DecisionTreeClassifier(criterion='gini', max_depth=2)

    elif model == 'forest':
        module = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='gini')

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
    # module = 'GBDT'         # offline: 0.0826341432788 #
    module = 'xgboost'      # offline: 0.0827897548881 #
    # module = 'lightGBM'     # offline: 0.0825307131425 #
    # module = 'tree'         # offline: 0.0843034628994 #
    # module = 'forest'       # offline: 0.0833564894226 #
    drill_module(model=module, is_store=False)                # is_store:是否保存当前结果 #

    if False:
        result = pd.read_csv('result.csv', low_memory=True)
        print(result[result['predicted_score'] > 0.1])
        print(result[result['predicted_score'] > 0.1].shape)

if __name__ == '__main__':
    main()

