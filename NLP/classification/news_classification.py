#coding=utf-8

import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn import cross_validation,metrics
from sklearn.ensemble import GradientBoostingClassifier

'''载入自定义词典和停用词'''
jieba.load_userdict(r'D:\dataSet\NLP\update_data\userdict.txt')
# jieba.load_userdict(r'D:\dataSet\NLP\update_data\dict.txt')
jieba.analyse.set_stop_words(r'D:\dataSet\NLP\update_data\stopped_words_two.txt')
jieba.analyse.set_stop_words(r'D:\dataSet\NLP\update_data\stopped_words_one.txt')

'''自定义函数：去除网页标签元素'''
def delete(string):
    string = string.replace("<", "|")
    string = string.replace(">", "|")
    string = string.split("|")
    tmp = ''
    for i in range(len(string)):
        if i % 2 == 0:
            tmp += string[i]
    return tmp

# '''读取数据'''
# read_data = pd.read_csv(r'D:\dataSet\NLP\news_classification\effective_news.csv', header=None, encoding='utf-8')   #  , nrows=100  #
# test_data = pd.read_csv(r'D:\dataSet\NLP\news_classification\not_passed_news.csv', encoding='utf-8')   #  , nrows=100  #
# read_data.columns = ['content']
# test_data.columns = ['1', 'content', '2', '3', '4', '5', '6', '7', '8']
#
# # print(test_data.shape[0])
# # print(test_data[test_data['1'].notnull()].shape[0])
# # print(test_data[test_data['content'].notnull()].shape[0])
# # print(test_data[test_data['2'].notnull()].shape[0])
# # print(test_data[test_data['3'].notnull()].shape[0])
# # print(test_data[test_data['4'].notnull()].shape[0])
# # print(test_data[test_data['5'].notnull()].shape[0])
# # print(test_data[test_data['6'].notnull()].shape[0])
# # print(test_data[test_data['7'].notnull()].shape[0])
# # print(test_data[test_data['8'].notnull()].shape[0])
#
# # read_data:9570 #
# # test_data:28028 #
# # 去重复行 #
# # read_data:9549 #
# # test_data:18136
#
# read_data = read_data.drop_duplicates(['content'])
# test_data = test_data.drop_duplicates(['content'])
# test_data = test_data.iloc[:, [1]]
# read_data = read_data[read_data['content'].notnull()]
# test_data = test_data[test_data['content'].notnull()]
#
# read_data['img'] = [1 if string.find('img') != -1 else 0 for string in read_data['content']]
# test_data['img'] = [1 if string.find('img') != -1 else 0 for string in test_data['content']]
#
# read_data['content'] = [delete(string) for string in read_data['content']]
# test_data['content'] = [delete(string) for string in test_data['content']]
# read_data['label'] = [1 for index in range(len(read_data))]
# test_data['label'] = [0 for index in range(len(test_data))]
#
# read_data = read_data[read_data['content'].notnull()]
# test_data = test_data[test_data['content'].notnull()]
# read_data['tf-idf'] = [','.join(jieba.analyse.extract_tags(string, withWeight=False)) for string in read_data['content']]
# test_data['tf-idf'] = [','.join(jieba.analyse.extract_tags(string, withWeight=False)) for string in test_data['content']]
# # read_data.to_csv('positive_sample.csv', index=None, encoding='utf-8')
# # test_data.to_csv('negative_sample.csv', index=None, encoding='utf-8')
# print(read_data)
# print(test_data)
# # print(test_data[test_data['label'].isnull()])


#########################################################################
# positive_sample = pd.read_csv(r'positive_sample.csv', encoding='utf-8')
# negative_sample = pd.read_csv(r'negative_sample.csv', encoding='utf-8')
# positive_sample['label'] = positive_sample['label'].fillna(1)
# negative_sample['label'] = negative_sample['label'].fillna(0)
# test_sample = positive_sample.copy()
# test_sample = test_sample.append(negative_sample)
# test_sample = test_sample.fillna('')
#
# train_positive_sample = positive_sample.sample(n=1000, replace=True)
# train_negative_sample = negative_sample.sample(n=4000, replace=True)
# train_sample = train_positive_sample.copy()
# train_sample = train_sample.append(train_negative_sample)
# train_sample = train_sample.fillna('')
# # print(train_sample)
########################################################################
# 数据集1：1切分 #
#######################################################################
positive_sample = pd.read_csv(r'positive_sample.csv', encoding='utf-8')
negative_sample = pd.read_csv(r'negative_sample.csv', encoding='utf-8')
# positive_sample['label'] = positive_sample['label'].fillna(1)
# negative_sample['label'] = negative_sample['label'].fillna(0)
positive_sample_len = int(len(positive_sample) / 2)
negative_sample_len = int(len(negative_sample) / 2)
# 训练集 #
train_sample = positive_sample.iloc[0:positive_sample_len]
train_sample = train_sample.append(negative_sample.iloc[0:negative_sample_len])
# 测试集 #
test_sample = positive_sample.iloc[positive_sample_len:-1]
test_sample = test_sample.append(negative_sample.iloc[negative_sample_len:-1])
train_sample = train_sample.fillna(' ')
test_sample = test_sample.fillna(' ')
# print(train_sample)
# print(test_sample)
# print(train_sample[train_sample['label'].isnull()])  # content #
# print(test_sample[test_sample['label'].isnull()])    # label #
############################################################################
'''统计字符串中符号的数量'''
def count_symbol(string, flag):
    count = 0
    if flag == 0:
        for index in string:
            if index == '，':
                count += 1
    elif flag == 1:
        for index in string:
            if index == '。':
                count += 1
    elif flag == 2:
        for index in string:
            if index == '“' or index == '”':
                count += 1

    # 回车符号 -- 强特征 #
    elif flag == 3:
        for index in string:
            if index == '\n':
                count += 1
    return count

def get_feature(sample):

    sample['lenght'] = [len(string) for string in sample['content']]
    sample['douhao'] = [count_symbol(string, 0) for string in sample['content']]
    sample['juhao'] = [count_symbol(string, 1) for string in sample['content']]
    sample['maohao'] = [count_symbol(string, 2) for string in sample['content']]
    sample['enter'] = [count_symbol(string, 3) for string in sample['content']]
    sample['word_count'] = [len(string.split(',')) for string in sample['tf-idf']]
    sample['rate'] = sample['word_count'] / sample['lenght']
    sample['img_feature'] = [1 if index != 0 else 0 for index in sample['img']]
    # print(type(sample['img_feature'][0]))
    return sample

def train_module(train_sample, test_sample):
    train_feature = get_feature(train_sample)
    train_label = train_sample.iloc[:, [2]]            # 添加img后改为2 #
    test_feature = get_feature(test_sample)
    length = train_feature.shape[1]
    train_feature = train_feature.iloc[:, 4:length]    # 分词、添加img后改为3 #
    test_feature = test_feature.iloc[:, 4:length]      # 分词、添加img后改为3 #
    print(train_feature)
    print(train_label)
    print(test_feature)

    # LightGBM model #
    module = LGBMClassifier(
        task='train',
        boosting_type='gbdt',
        objective='binary',
        metric={'l2', 'auc'},
        min_data=1
    )
    module.fit(train_feature.values, train_label.values)
    proba = module.predict(test_feature.values)  # [:, 1] #

    # 正确预测出的垃圾新闻数: 8616
    # 算法输出的垃圾新闻数: 9965
    # 数据集中原本的垃圾新闻数: 9067
    # P: 0.864626191670848
    # R: 0.9502591816477335
    # F1: 0.905422446406053

    # # xgboost model #
    # module = xgb.XGBClassifier(
    #     learning_rate=0.1,
    #     n_estimators=300,
    #     max_depth=4,
    #     gamma=0.1,
    #     subsample=0.7,
    #     objective='binary:logistic',
    #     nthread=5,
    #     scale_pos_weight=1,
    #     # seed=27
    # )
    # module.fit(train_feature.values, train_label.values)
    # proba = module.predict(test_feature.values)  # [:, 1] #

    # 正确预测出的垃圾新闻数: 8651
    # 算法输出的垃圾新闻数: 10070
    # 数据集中原本的垃圾新闻数: 9067
    # P: 0.8590863952333664
    # R: 0.9541193338480203
    # F1: 0.9041124523175

    # # GBDT  model#
    # module = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, subsample=0.7)
    # module.fit(train_feature, train_label)
    # proba = module.predict(test_feature)

    # 正确预测出的垃圾新闻数: 8702
    # 算法输出的垃圾新闻数: 10135
    # 数据集中原本的垃圾新闻数: 9067
    # P: 0.8586087814504193
    # R: 0.9597441270541525
    # F1: 0.9063639204249557

    proba = pd.DataFrame(proba)
    proba.columns = ['probability']
    test_sample['probability'] = [int(index) for index in proba['probability']]
    test_sample = test_sample.iloc[:, [2, -1]]             # 添加img后改为2 #
    # test_sample.to_csv('result.csv', index=None)
    print(test_sample)
    return test_sample

def calculate_F():
    result = train_module(train_sample, test_sample)
    # result = pd.read_csv(r'result.csv')
    label = list(result['label'])
    probability = list(result['probability'])
    cal_p_down = 0  # Precision分母 #
    cal_up = 0  # Precision and Recall分子 #
    cal_r_down = 0  # Recall分母 #
    for index in range(len(label)):
        if probability[index] == 0.0:
            cal_p_down += 1
        if probability[index] == 0.0 and label[index] == 0.0:
            cal_up += 1
        if label[index] == 0.0:
            cal_r_down += 1

    p = cal_up / cal_p_down
    r = cal_up / cal_r_down
    print('\n正确预测出的垃圾新闻数:', cal_up)
    print('算法输出的垃圾新闻数:', cal_p_down)
    print('数据集中原本的垃圾新闻数:', cal_r_down)
    print('P:', p)
    print('R:', r)
    print('F1:', (2 * p * r) / (p + r))
    # auc = metrics.roc_auc_score(label, probability)  # 验证集上的auc值 #
    # print('AUC:', auc)

calculate_F()


