# -*-: coding=utf-8 :-*-

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


'''如果数据划分方式改变那要重新跑一遍'''
user_register = pd.read_csv(r'../../dataset/user_register.csv', encoding='utf-8')
train_feature = pd.read_csv(r'train_feature.csv', encoding='utf-8')
validate_feature = pd.read_csv(r'validate_feature.csv', encoding='utf-8')
test_feature = pd.read_csv(r'test_feature.csv', encoding='utf-8')

train_feature = train_feature.append(validate_feature)
train_feature.index = range(len(train_feature))

train_feature = train_feature[['user_id', 'label']]
train_feature = train_feature.merge(user_register[['user_id', 'register_type', 'device_type']], on='user_id', how='left')
test_feature = test_feature[['user_id']].merge(user_register[['user_id', 'register_type', 'device_type']], on='user_id', how='left')
print(train_feature.sort_values(by=['user_id']))
user_train = train_feature[['user_id']]   # 用户ID #
user_test = test_feature[['user_id']]     # 用户ID #
test_feature = test_feature[['register_type', 'device_type']]


label = train_feature[['label']]
features = train_feature[['register_type', 'device_type']]
model = OneHotEncoder()
features = model.fit_transform(features).toarray()
# print(pd.DataFrame(features))

test = model.transform(test_feature).toarray()
# print(test)

module = LogisticRegression(penalty='l2', solver='sag', max_iter=500, random_state=42, n_jobs=4)  # , solver='sag'
module.fit(features, label)

train_predict = module.predict_proba(features)[:, 1]
test_predict = module.predict_proba(test)[:, 1]

user_train['predict'] = pd.Series(train_predict)
user_test['predict'] = pd.Series(test_predict)

print(user_train.sort_values(by=['user_id']))
print(user_test)

user_test.to_csv(r'user_device_type_prob.csv', encoding='utf-8', index=None)

