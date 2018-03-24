#coding=utf-8

'''
Date:2018-03-23
Admin：roger
log_loss_online: 0.08328
'''

import pandas as pd
import numpy as np
import time
import os
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier      #Gradient Boosting 和 AdaBoost算法#
import xgboost as xgb



'''读取数据集'''
ijcai_train_data = pd.read_csv('../data_pre_deal/ijcai_train_data.csv', low_memory=False)    # , nrows=1000 #
ijcai_test_data = pd.read_csv('../data_pre_deal/ijcai_test_data.csv', low_memory=False)

ijcai_train_data['context_timestamp'] = [time.localtime(index) for index in ijcai_train_data['context_timestamp']]
ijcai_train_data['context_timestamp'] = [time.strftime("%Y%m%d", index) for index in ijcai_train_data['context_timestamp']]
ijcai_train_data = ijcai_train_data.drop_duplicates(['instance_id'])   # 数据集去重 #

'''去噪声值'''
ijcai_train_data['shop_review_positive_rate'] = [index if index != -1 else ijcai_train_data['shop_review_positive_rate'].mean() for index in ijcai_train_data['shop_review_positive_rate']]
ijcai_train_data['shop_score_service'] = [index if index != -1 else ijcai_train_data['shop_score_service'].mean() for index in ijcai_train_data['shop_score_service']]
ijcai_train_data['shop_score_delivery'] = [index if index != -1 else ijcai_train_data['shop_score_delivery'].mean() for index in ijcai_train_data['shop_score_delivery']]
ijcai_train_data['shop_score_description'] = [index if index != -1 else ijcai_train_data['shop_score_description'].mean() for index in ijcai_train_data['shop_score_description']]


validate_data = ijcai_train_data[(ijcai_train_data['context_timestamp'] >= '20180918') & (ijcai_train_data['context_timestamp'] <= '20180919')]
train_data = ijcai_train_data[(ijcai_train_data['context_timestamp'] >= '20180920') & (ijcai_train_data['context_timestamp'] <= '20180922')]
test_data = ijcai_train_data[(ijcai_train_data['context_timestamp'] >= '20180923') & (ijcai_train_data['context_timestamp'] <= '20180924')]

test_data = test_data.iloc[:, 0: -1]
train_label = train_data[['is_trade']]
test_data = test_data.append(ijcai_test_data)

# print(test_data)
# print(train_label)
'''user_id标签'''
user_gender_id = list(ijcai_train_data.drop_duplicates(['user_gender_id'])['user_gender_id'])
ijcai_train_data['user_occupation_id'] = [-2 if index == -1 else index for index in ijcai_train_data['user_occupation_id']]
user_occupation_id = list(ijcai_train_data.drop_duplicates(['user_occupation_id'])['user_occupation_id'])
ijcai_train_data['user_star_level'] = [-3 if index == -1 else index for index in ijcai_train_data['user_star_level']]
user_star_level = list(ijcai_train_data.drop_duplicates(['user_star_level'])['user_star_level'])

'''shop_id标签'''
def add_shop(List):
    for index in range(len(List)):
        List[index] = List[index] + 10000
    return List
shop_review_num_level = add_shop(list(ijcai_train_data.drop_duplicates(['shop_review_num_level'])['shop_review_num_level']))
List = []
shop_review_num_level = sorted(shop_review_num_level)
List.append(shop_review_num_level[0])
List.append(shop_review_num_level[-1])
shop_review_num_level = List.copy()
shop_star_level = add_shop(list(ijcai_train_data.drop_duplicates(['shop_star_level'])['shop_star_level']))
List = []
shop_star_level = sorted(shop_star_level)
List.append(shop_star_level[0])
List.append(shop_star_level[-1])
shop_star_level = List.copy()
# print(shop_review_num_level, shop_star_level)

'''特征提取函数'''
def extract_feature(dataset):
    dataset.index = range(dataset.shape[0])

    '''元数据特征'''
    feature = dataset.loc[:, ['instance_id', 'item_price_level']]
    feature['item_collected_level'] = dataset['item_collected_level']   # [index for index in dataset['item_collected_level']]
    feature['item_pv_level'] = [index for index in dataset['item_pv_level']]

    feature['shop_review_num_level'] = [index for index in dataset['shop_review_num_level']]
    feature['shop_review_positive_rate'] = [index for index in dataset['shop_review_positive_rate']]
    feature['shop_star_level'] = [index - 4998 for index in dataset['shop_star_level']]
    feature['shop_score_service'] = [index for index in dataset['shop_score_service']]
    feature['shop_score_delivery'] = [index for index in dataset['shop_score_delivery']]
    feature['shop_score_description'] = [index for index in dataset['shop_score_description']]

    '''item系列'''
    dataset['item_category_list_str'] = [index.split(';') for index in dataset['item_category_list']]  # 按;切片 #
    dataset['item_category_list_len'] = [len(index) for index in dataset['item_category_list_str']]  # 切片后字符串长度 #
    dataset['item_category_list_two'] = [index[1] for index in dataset['item_category_list_str']]

    item_category_list_one_two = ['8277336076276184272', '5755694407684602296', '4879721024980945592', '2011981573061447208',
                                  '7258015885215914736', '509660095530134768', '8710739180200009128', '5799347067982556520',
                                  '2642175453151805566', '2436715285093487584', '3203673979138763595', '22731265849056483',
                                  '1968056100269760729']

    for index in item_category_list_one_two:
        feature[index] = [1 if (index == string) else 0 for string in dataset['item_category_list_two']]
    feature['item_category_list_len'] = [index for index in dataset['item_category_list_len']]

    '''item_property_list'''
    dataset['item_property_list_str'] = [index.split(';') for index in dataset['item_property_list']]
    dataset['item_property_list_len'] = [len(index) for index in dataset['item_property_list_str']]
    max_mun = dataset['item_property_list_len'].max()
    min_num = dataset['item_property_list_len'].min()
    dataset['item_property_list_num'] = [(index - min_num) / (max_mun - min_num) for index in dataset['item_property_list_len']]
    feature['item_property_list_num'] = [index for index in dataset['item_property_list_num']]
    '''item_sales_level'''
    dataset['item_sales_level'] = [index if index != -1 else dataset['item_sales_level'].median() for
                                            index in dataset['item_sales_level']]
    feature['item_sales_level'] = [index for index in dataset['item_sales_level']]

    '''user_id series系列'''
    '''user_age_level'''
    dataset['user_age_level'] = [(index - 1000) if index >= 1000 else (dataset['user_age_level'].mode()[0] - 1000) for
                                 index in dataset['user_age_level']]
    feature['user_age_level'] = [index for index in dataset['user_age_level']]
    '''user_gender_id'''
    for index in user_gender_id:
        feature[index] = [1 if string == index else 0 for string in dataset['user_gender_id']]
    '''user_occupation_id'''
    for index in user_occupation_id:
        feature[index] = [1 if string == index else 0 for string in dataset['user_occupation_id']]
    '''user_star_level'''
    for index in user_star_level:
        feature[index] = [1 if string == index else 0 for string in dataset['user_star_level']]
    '''user统计特征'''
    user_count = pd.pivot_table(dataset, index='user_id', values='user_gender_id', aggfunc='count')
    user_count['user_id'] = user_count.index
    user_count.columns = ['user_count', 'user_id']
    dataset = dataset.merge(user_count, on='user_id', how='left')
    feature['user_count'] = [index for index in dataset['user_count']]
    feature['user_count_two'] = [1 if string < 3 else 0 for string in feature['user_count']]
    feature['user_count_three'] = [1 if string > 6 else 0 for string in feature['user_count']]  # 0.0929142537179 GBDT #

    '''shop_id Series系列'''
    '''shop_review_num_level shop_star_level'''
    '''取最大最小试试'''
    for index in shop_review_num_level:
        feature[index] = [1 if ((string+10000) == index) else 0 for string in dataset['shop_review_num_level']]
    for index in shop_star_level:
        feature[index] = [1 if ((string+10000) == index) else 0 for string in dataset['shop_star_level']]
    shop_count = pd.pivot_table(dataset, index='shop_id', values='shop_review_num_level', aggfunc='count')
    shop_count['shop_id'] = shop_count.index
    shop_count.columns = ['shop_count', 'shop_id']
    dataset = dataset.merge(shop_count, on='shop_id', how='left')
    feature['shop_count'] = [index for index in dataset['shop_count']]

    max_count = feature['shop_count'].max()
    min_count = feature['shop_count'].min()
    feature['shop_count_attribute_one'] = [(index - min_count) / (max_count - min_count) for index in feature['shop_count']]
    feature['shop_count_two'] = [1 if (string < 9700) & (string > 3000) else 0 for string in feature['shop_count']]
    feature['shop_count_three'] = [1 if string < 100 else 0 for string in feature['shop_count']]  # 0.0928658768241 GBDT #
    # feature['total'] = feature[['shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
    #      'shop_score_delivery', 'shop_score_description']].apply(lambda x: 0.05*x.shop_review_num_level + 0.3*x.shop_review_positive_rate + 0.05*x.shop_star_level + 0.2*x.shop_score_service + 0.2*x.shop_score_delivery + 0.2*x.shop_score_description, axis=1)

    # 0.0928800084408
    print(feature.iloc[:, 1:])
    return feature

train_feature = extract_feature(train_data).iloc[:, 1:]

test_feature = ijcai_test_data.loc[:, ['instance_id']]
test_feature = test_feature.merge(extract_feature(test_data), on='instance_id', how='left').iloc[:, 1:]            # test_data #

'''GBDT module'''
module = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=2)
module.fit(train_feature, train_label)              # 0.0969705847886  0.0955247060859 0.095190866353 0.0949605724075#
result = module.predict_proba(test_feature)[:, 1]

# '''xgboost module'''
# module = xgb.XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=200,
#         max_depth=2,
#         gamma=0.1,
#         subsample=0.7,
#         objective='reg:linear',
#         nthread=4,
#     )
# module.fit(train_feature, train_label)         # 200/2 0.094978499278 #
# result = module.predict_proba(test_feature)[:, 1]

# joblib.dump(module, "model.m")    # 保存模型 #

# module = joblib.load("model.m")   # 读取模型 #
# result = module.predict_proba(test_feature)[:, 1]

result = pd.DataFrame(result)
result.columns = ['predicted_score']
sample = ijcai_test_data.loc[:, ['instance_id']]
sample['predicted_score'] = [index for index in result['predicted_score']]
print(sample)
# sample.to_csv('result.csv', index=None)


'''模型验证'''
validate_feature = extract_feature(validate_data).iloc[:, 1:]
validate_label = validate_data[['is_trade']]

# module = joblib.load("model.m")   # 读取模型 #

validate_label_predict = module.predict_proba(validate_feature)[:, 1]
validate_label_predict = pd.DataFrame(validate_label_predict)
print(validate_label_predict)

score = log_loss(validate_label, validate_label_predict)
print('log_loss_score:', score)


# 50 0.0929142537179
# 97 0.092977776133

# f = open("result.csv", 'r')             # 返回一个文件对象
# r = open("result.txt", 'w')
# line = f.readline()             # 调用文件的 readline()方法
# while line:
#     line = line.replace(',', ' ')
#     r.write(line)
#     # print(line, end='')    # 在 Python 3中使用#
#     line = f.readline()
# f.close()
# r.close()


# result = pd.read_csv('result.csv', low_memory=False)
# print(result[result['predicted_score'] > 0.2])

