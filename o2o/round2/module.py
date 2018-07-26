# -*- coding:utf-8 -*-

import time
import xgboost as xgb
from extract_features import *

result_path = 'dataSet//'
submit_path = 'submit//'


def calc_auc_score(data_frame, predict):
    '''
    :param data_frame:['Coupon_id', 'label']
    :param predict: predict = []
    :return:auc score
    '''
    def return_score(label, predict):
        if len(label) == 1:
            return 0
        if len(set(label)) == 1:
            return 0

        # if len(set(label)) == 1:
        #     if 0 in set(label):
        #         return 0
        #     else:
        #         return 1

        return roc_auc_score(label, predict)

    data_frame['predict'] = [index for index in predict]
    data_frame_label = pd.pivot_table(data_frame, index='Coupon_id', values=['label'], aggfunc=return_list).reset_index()
    data_frame_predict = pd.pivot_table(data_frame, index='Coupon_id', values=['predict'], aggfunc=return_list).reset_index()
    data_frame_label = data_frame_label.merge(data_frame_predict, on='Coupon_id', how='left')
    # data_frame_label.to_csv(submit_path + 'result.csv', index=None, encoding='utf-8')
    # print('store ok.')
    data_frame_label['score'] = data_frame_label.apply(lambda index: return_score(index.label, index.predict), axis=1)
    print(data_frame_label)
    return np.mean(data_frame_label['score'])


def train_xgb_module(store_features=False, store_result=False, select_feature=False, num_round=500):
    if store_features is True:
        '''feature'''
        print('start extract label feature section features')
        train_feature = extract_label_section_features('2016-04-01', '2016-05-01', ccf_off_train_data)
        validate_feature = extract_label_section_features('2016-05-15', '2016-06-15', ccf_off_train_data)
        test_feature = extract_label_section_features('2016-07-01', '2016-07-32', ccf_off_test_data)
        print('feature label features extract successfully!')

        print('start extract feature section features')
        train_feature = extract_feature_section_features('2016-01-15', '2016-03-15', ccf_off_train_data, train_feature)
        validate_feature = extract_feature_section_features('2016-03-15', '2016-05-01', ccf_off_train_data, validate_feature)
        test_feature = extract_feature_section_features('2016-05-01', '2016-06-15', ccf_off_train_data, test_feature)
        print('feature section features extract successfully!')
        '''store'''
        try:
            print('start to store features...')
            train_feature.to_csv(result_path + 'train_feature.csv', encoding='utf-8', index=None)
            validate_feature.to_csv(result_path + 'validate_feature.csv', encoding='utf-8', index=None)
            test_feature.to_csv(result_path + 'test_feature.csv', encoding='utf-8', index=None)
            print('store features successfully!')
        except:
            print('store features failed...')
    else:
        train_feature = pd.read_csv(result_path + 'train_feature.csv', encoding='utf-8', low_memory=False)
        validate_feature = pd.read_csv(result_path + 'validate_feature.csv', encoding='utf-8', low_memory=False)
        test_feature = pd.read_csv(result_path + 'test_feature.csv', encoding='utf-8', low_memory=False)
        print('read features successfully!')

    train_label = train_feature[['label']]
    validate_label = validate_feature[['label']]

    #########
    validate_data_frame = validate_feature[['Coupon_id']]
    validate_data_frame['label'] = np.array(validate_label['label'])

    train_feature = train_feature.drop(['label'], axis=1)
    validate_feature = validate_feature.drop(['label'], axis=1)

    if select_feature is True:
        features_name = []
        train_feature = train_feature[features_name]
        validate_feature = validate_feature[features_name]
        test_feature = test_feature[features_name]
        print('feature\'s num:', len(features_name))
    else:
        train_feature = train_feature.drop(['Merchant_id', 'Coupon_id', 'User_id', 'Date'], axis=1)
        validate_feature = validate_feature.drop(['Merchant_id', 'Coupon_id', 'User_id', 'Date'], axis=1)
        test_feature = test_feature.drop(['Merchant_id', 'Coupon_id', 'User_id'], axis=1)

    train_test_feature = train_feature.append(validate_feature)  # train_feature.append(validate_feature)
    train_test_label = train_label.append(validate_label)  # train_label.append(validate_label)

    params = {
        'booster': 'gbtree',
        'max_depth': 4,  # 5 #
        'colsample': 0.8,
        'subsample': 0.8,
        'eta': 0.02,     # 0.03 #
        'silent': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'min_child_weight': 5,
        'scale_pos_weight': 1,
        'nthread ': 6,
    }

    '''参数搜索
    xgb_module = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        objective='binary:logistic',
        subsample=0.8,
        learning_rate=0.03,
        # booster='gbtree',
        nthread=5,
        min_child_weight=5,
        colsample_bytree=0.8,
        scale_pos_weight=1,
    )

    # website: https://blog.csdn.net/u012969412/article/details/72973055 #
    params = {"n_estimators": np.arange(400, 1000, 100), 'learning_rate': np.arange(0.02, 0.1, 0.01), 'max_depth': np.arange(4, 8, 1)}
    print(params)
    grid = GridSearchCV(estimator=xgb_module, param_grid=params, scoring='roc_auc', cv=3, verbose=2)
    grid.fit(train_feature, train_label['label'].ravel())
    print(grid.best_score_)
    # print(grid.best_estimator_.n_neighbors)
    print(grid.best_params_)
    exit(0)
    '''

    validate = xgb.DMatrix(train_feature, label=train_label)
    validate_feature = xgb.DMatrix(validate_feature)
    module = xgb.train(params, validate, num_round)
    predict = module.predict(validate_feature)
    print('iteration round:', num_round)
    # print('auc score:', roc_auc_score(validate_label['label'], predict))
    print('auc score:', calc_auc_score(validate_data_frame, predict))

    del train_feature
    del validate_feature
    gc.collect()

    if store_result is True:
        time_string = time.strftime('_%Y%m%d%H%M%S', time.localtime(time.time()))
        file_name = 'submit' + time_string + '.csv'

        train = xgb.DMatrix(train_test_feature, label=train_test_label)
        test_feature = xgb.DMatrix(test_feature)
        module_two = xgb.train(params, train, num_round)
        result = module_two.predict(test_feature)
        test_user = pd.read_csv(result_path + 'ccf_offline_stage1_test_revised.csv', encoding='utf-8')

        test_user = test_user[['User_id', 'Coupon_id', 'Date_received']]
        test_user['Probability'] = [index for index in result]
        print(test_user)
        test_user.to_csv(submit_path + file_name, index=None, encoding='utf-8', header=None)
        print('%s store successfully!' % file_name)

    print('program is over!')


if __name__ == '__main__':
    start_time = time.clock()
    train_xgb_module(store_features=True, store_result=False, select_feature=False, num_round=500)
    end_time = time.clock()
    print('program spend time：', end_time - start_time, ' sec')

