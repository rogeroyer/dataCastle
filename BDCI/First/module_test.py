#coding=utf-8
import pandas as pd
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

# train_data = pd.read_csv(r'train_feature.csv', nrows=600000)#, nrows=500000
# label_data = pd.read_csv(r'label_data.csv', nrows=600000)#, nrows=500000
# test_data = pd.read_csv(r'test_feature.csv')  #, nrows=5000000
# train_data = train_data.iloc[:, 1:]
# test_data = test_data.iloc[:, 2:]
# train_data.columns = [index for index in range(68)]  # 68随着特征维数不同而不同 #
# label_data.columns = [0]
# test_data.columns = [index for index in range(68)]
# print(train_data)
# print(label_data)
# print(test_data)


test_data = pd.read_csv(r'test_data.csv')
print(test_data)


clf = xgb.XGBClassifier(
    learning_rate=0.3,
    n_estimators=250,
    max_depth=5,
    gamma=0.1,
    subsample=0.7,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)

# clf.fit(train_data.values, label_data.values)
# proba = clf.predict_proba(test_data.values)[:, 1]
# proba = pd.DataFrame(proba)
# proba.columns = ['probability']
# proba.to_csv('proba.csv', index=None)
# print(proba)




'''筛选每个row_id概率最大的shop_id'''
# test_data = pd.read_csv(r'test_feature.csv')  #, nrows=5000000
# proba = pd.read_csv(r'proba.csv')
# test_data = test_data.iloc[:, [0, 1]]
# test_data['probability'] = proba['probability']
# # test_data['rank'] = test_data['probability'].rank()
# # test_data = test_data.sort_values(by=['probability'])
# test_data = test_data.sort_values(by=['row_id', 'probability'], ascending=False) #核心步骤#
# print(test_data)
# test_data = test_data.drop_duplicates(['row_id']) #核心步骤#
#
# test_data = test_data.iloc[:, [0, 1]]
# print(test_data)
# test_data.to_csv('result.csv', header=None, index=None)
# print('Finished!!')


'''暂时无用操作'''
# sub_sample = pd.pivot_table(test_data, index='row_id', values=['shop_id', 'probability'], aggfunc='max')
# sub_sample = pd.pivot_table(test_data, index=['shop_id', 'row_id'], values='probability', aggfunc='max')
# sub_sample = pd.pivot_table(test_data, index='row_id', values='probability', aggfunc='max')
# sub_sample = pd.pivot_table(test_data, index='row_id', values='probability', aggfunc=cal_max_shop)
# sub_sample_two = pd.pivot_table(test_data, index='row_id', values=['shop_id', 'probability'], aggfunc=cal_max_shop)
# sub_sample['row_id'] = sub_sample.index
# sub_sample_two['row_id'] = sub_sample_two.index
# sub_sample.merge(test_data.iloc[:, [1, 2]], on='probability', how='left')
# sub_sample = pd.pivot_table(test_data, index='row_id', values='probability', aggfunc=cal_max_shop)
# print(sub_sample)
# print(sub_sample_two)

'''填补空缺的row_id'''
# read_data_two = pd.read_csv(r'D:\dataSet\aliyun_ccf\ABtest-evaluation_public.csv')
# read_data_two = read_data_two.iloc[:, [0]].drop_duplicates(['row_id'])
# sub_sample = pd.read_csv(r'result.csv')
# sub_sample.columns = ['row_id', 'shop_id']
# read_data_two = read_data_two.merge(sub_sample, on='row_id', how='left')
# read_data_two['shop_id'] = read_data_two['shop_id'].fillna(read_data_two['shop_id'][1])
# print(read_data_two[read_data_two['shop_id'].isnull()])
# read_data_two.to_csv('result.csv', index=None)
# print(read_data_two)

result = 0.1235
