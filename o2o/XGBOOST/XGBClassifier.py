#coding-utf-8
import xgboost as xgb

clf = xgb.XGBClassifier(
    base_score=0.5, 
    colsample_bylevel=1, 
    colsample_bytree=0.8, 
    gamma=0, 
    learning_rate=0.1, 
    max_delta_step=0, 
    max_depth=4,
    min_child_weight=6, 
    missing=None, 
    n_estimators=140, 
    nthread=4,
    objective='binary:logistic', 
    reg_alpha=0, reg_lambda=1,
    scale_pos_weight=1, seed=27, 
    silent=True, subsample=0.8
)

# XGBClassifier参数调优还在进一步修改中 #

clf.fit(train_data, label_data)
proba = clf.predictproba(test_data)[:, 1]

trian_data # 训练集特征 #
label_data # 训练集标签 #
test_data  # 测试集特征 #
