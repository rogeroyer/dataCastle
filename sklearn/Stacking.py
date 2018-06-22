# --* coding:utf-8 *--

'''
Create by roger.
Date:2018.06
'''

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
# from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


xgb_module = xgb.XGBClassifier(
                booster='gbtree',
                learning_rate=0.02,       # 0.02 #
                n_estimators=500,         # 500 #
                max_depth=6,              # 4
                gamma=0.1,
                subsample=0.8,            # 0.7
                objective='binary:logistic',
                n_jobs=4,
                min_child_weight=16,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                reg_alpha=0.01
            )


lgb_module = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            num_leaves=96,   # num_leaves = 2^max_depth * 0.6 #
            max_depth=5,
            n_estimators=500,
            learning_rate=0.05,
            n_jobs=4
        )

lr_module = LogisticRegression(
    penalty='l2',
    solver='sag',
    max_iter=500,
    random_state=42,
    n_jobs=4

)

rf_module = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    criterion='gini',
    n_jobs=4,
    random_state=2018
)

gb_module = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=2018
)


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2018))   # len(y),  , n_splits=self.n_folds
        s_train = np.zeros((X.shape[0], len(self.base_models)))
        s_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            print(i, clf)
            s_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                # clf.fit(X_train, y_train)
                clf.fit(X_train, y_train.ravel())
                y_pred = clf.predict_proba(X_holdout)[:, 1][:]
                s_train[test_idx, i] = y_pred
                s_test_i[:, j] = clf.predict_proba(T)[:, 1][:]
            s_test[:, i] = s_test_i.mean(1)
        self.stacker.fit(s_train, y.ravel())
        y_pred = self.stacker.predict_proba(s_test)[:, 1][:]
        y_pred = pd.DataFrame(y_pred, columns=['PROB'])
        # print(y_pred)
        return y_pred


    def read_data(self):
        test = pd.read_csv(r'test_feature.csv', encoding='utf-8', nrows=None)    ##
        train_test = pd.read_csv(r'train_feature.csv', encoding='utf-8', nrows=None)    ##
        label = pd.read_csv(r'train_label.csv', encoding='utf-8', nrows=None)    ##
        print('读取数据完毕。。。')

        train_test = train_test.iloc[:, 1:]
        result = test[['instance_id']]
        test = test.iloc[:, 1:]
        return train_test, label, test, result


if __name__ == '__main__':
    ensemble = Ensemble(4, lr_module, [xgb_module, lgb_module, rf_module, gb_module])
    train_test, label, test, sample = ensemble.read_data()
    result = ensemble.fit_predict(train_test, label, test)
    print('模型融合完毕。。。')
    result = pd.DataFrame(result, columns=['PROB'])

    sample['predicted_score'] = [index for index in result['PROB']]
    print(sample)
    sample.to_csv(r'stacking.csv', index=None)
    print('数据整合完毕。。。')

    f = open("stacking.csv", 'r')  # 返回一个文件对象
    r = open('stacking.txt', 'w')
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        line = line.replace(',', ' ')
        r.write(line)
        line = f.readline()
    f.close()
    r.close()
    print('The result have updated!')

