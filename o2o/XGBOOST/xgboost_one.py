# coding=utf-8

import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

train_data = pd.read_csv(r'predict_data.csv', header=None)
label_data = pd.read_csv(r'predict_label.csv', header=None)
test_data = pd.read_csv(r'predict_test.csv', header=None)
sub_sample = pd.read_csv(r'D:\aliyun\ccf_offline_stage1_test_revised.csv', header=None)
sub_sample.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
sub_sample = sub_sample.iloc[:, [0, 2, 5]]
# train_data = xgb.DMatrix(train_data)
# label_data = xgb.DMatrix(label_data)
# test_data = xgb.DMatrix(test_data)

# print(train_data)
# print(label_data)
# print(test_data)
# print(sub_sample)

# dtrain = xgb.DMatrix(train_data, label=label_data)
# dtest = xgb.DMatrix(test_data)

# params = {
#             'booster': 'gbtree',
#             'objective': 'binary:logistic',
#             'eta': 0.1,
#             'max_depth': 10,
#             'subsample': 1.0,
#             'min_child_weight': 5,
#             'colsample_bytree': 0.2,
#             'scale_pos_weight': 0.1,
#             'eval_metric': 'auc',
#             'gamma': 0.2,
#             'lambda': 300
# }

clf = xgb.XGBClassifier(
    learning_rate=0.2,
    n_estimators=150,
    max_depth=5,
    gamma=0.1,
    subsample=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)

clf.fit(train_data, label_data)
# print(clf.predict(test_data))
# print('Hello Xgboost')
proba = clf.predict_proba(test_data)[:, 1]
proba = pd.DataFrame(proba)
proba.columns = ['probability']
print(proba)
sub_sample['probability'] = proba['probability']
sub_sample.to_csv('sample_submission.csv', header=None, index=None)
print(sub_sample)

auc=
