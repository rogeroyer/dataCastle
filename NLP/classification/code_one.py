#coding=utf-8

import pandas as pd
import jieba
import jieba.analyse
import jieba.posseg as pseg
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation,metrics

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
# # read_data:28028 #
# # test_data:9570 #
#
# test_data = test_data.iloc[:, [1]]
# test_data = test_data[test_data['content'].notnull()]
# read_data['content'] = [delete(string) for string in read_data['content']]
# test_data['content'] = [delete(string) for string in test_data['content']]
# read_data['label'] = [1 for index in range(len(read_data))]
# test_data['label'] = [0 for index in range(len(test_data))]
# test_data = test_data[test_data['content'].notnull()]
# read_data.to_csv('positive_sample.csv', index=None, encoding='utf-8')
# test_data.to_csv('negative_sample.csv', index=None, encoding='utf-8')
# print(test_data)
# print(read_data)


positive_sample = pd.read_csv(r'positive_sample.csv', encoding='utf-8')
negative_sample = pd.read_csv(r'negative_sample.csv', encoding='utf-8')
positive_sample['label'] = positive_sample['label'].fillna(1)
negative_sample['label'] = negative_sample['label'].fillna(0)
test_sample = positive_sample.copy()
test_sample = test_sample.append(negative_sample)
test_sample = test_sample.fillna('')

train_positive_sample = positive_sample.sample(n=1000, replace=True)
train_negative_sample = negative_sample.sample(n=4000, replace=True)
train_sample = train_positive_sample.copy()
train_sample = train_sample.append(train_negative_sample)
train_sample = train_sample.fillna('')
# print(train_sample)

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
    # print(sample)
    return sample

def train_module(train_sample, test_sample):
    train_feature = get_feature(train_sample)
    train_label = train_sample.iloc[:, [1]]
    test_feature = get_feature(test_sample)
    length = train_feature.shape[1]
    train_feature = train_feature.iloc[:, 2:length]
    test_feature = test_feature.iloc[:, 2:length]
    print(train_feature)
    print(train_label)
    print(test_feature)
    module = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=4,
        gamma=0.1,
        subsample=0.7,
        objective='binary:logistic',
        nthread=5,
        scale_pos_weight=1,
        seed=27
    )

    module.fit(train_feature.values, train_label.values)
    proba = module.predict(test_feature.values)  # [:, 1] #
    proba = pd.DataFrame(proba)
    proba.columns = ['probability']
    test_sample['probability'] = proba['probability']
    test_sample = test_sample.iloc[:, [1, -1]]
    test_sample.to_csv('result.csv', index=None)
    print(test_sample)

train_module(train_sample, test_sample)

result = pd.read_csv(r'result.csv')
label = list(result['label'])
probability = list(result['probability'])
count = 0
for index in range(len(label)):
    if label[index] == probability[index]:
        count += 1
r = p = count / len(label)
print('P：', p)
print('F-score:', (2 * p * r) / (p + r))

test_auc = metrics.roc_auc_score(label, probability)  #验证集上的auc值
print('AUC:', test_auc)




# read_data['content'] = [delete(string) for string in read_data['content']]
# read_data['textrank'] = [','.join(jieba.analyse.textrank(string, withWeight=False)) for string in read_data['content']]  # topK=5,  #
# read_data['tf-idf'] = [','.join(jieba.analyse.extract_tags(string, withWeight=False)) for string in read_data['content']]  # topK=5,   #
# print(read_data)
#
# test_data['content'] = [delete(string) for string in test_data['content']]
# test_data['textrank'] = [','.join(jieba.analyse.textrank(string, withWeight=False)) for string in test_data['content']]
# test_data['tf-idf'] = [','.join(jieba.analyse.extract_tags(string, withWeight=False)) for string in test_data['content']]
# print(test_data)

