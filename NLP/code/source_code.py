
#coding=utf-8

from time import clock
import pandas as pd
import jieba.analyse
import jieba.posseg as pseg
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

'''读取自定义词典和停用词'''
jieba.load_userdict(r'D:\dataSet\NLP\update_data\userdict.txt')
jieba.analyse.set_stop_words(r'D:\dataSet\NLP\update_data\stopped_words_two.txt')
jieba.analyse.set_stop_words(r'D:\dataSet\NLP\update_data\stopped_words_one.txt')

'''去除< **** >信息'''
def delete(string):
    string = string.replace("<", "|")
    string = string.replace(">", "|")
    string = string.split("|")
    tmp = ''
    for i in range(len(string)):
        if i % 2 == 0:
            tmp += string[i]
    return tmp

# '''测试程序运行时间'''
# start = clock()
# '''测试数据集'''
# read_data = pd.read_csv(r'D:\dataSet\NLP\update_data\news_50000.csv', encoding='utf-8', low_memory=False)    # , encoding="gb2312"  , nrows=10
# print(read_data)
# read_data = read_data.iloc[:, [0, 2]]
# # print(read_data)
#
# read_data['content'] = [str(string) for string in read_data['content']]
# read_data['content'] = [delete(string) for string in read_data['content']]
#
# # read_data['text_rank'] = [jieba.analyse.textrank(string, topK=3, withWeight=False) for string in read_data['content']]
# # read_data['key_words'] = [string[0]+','+string[1]+','+string[2] if len(string) == 3 else string for string in read_data['text_rank']]
# # read_data = read_data.iloc[:, [0, 1, 3]]
#
# read_data['textrank'] = [','.join(jieba.analyse.textrank(string, topK=3, withWeight=False)) for string in read_data['content']]
# read_data['tf-idf'] = [','.join(jieba.analyse.extract_tags(string, topK=3,  withWeight=False)) for string in read_data['content']]
#
# # read_data.to_csv('first_split.csv', index=None, encoding='utf-8')
# print(read_data)
#
# finish = clock()
# print((finish-start)/1000000)
# print('Finished!')

# test_data = pd.read_csv(r'first_split.csv', low_memory=False)
# print(test_data.iloc[:, [2, 3]])



#########################################################################

'''测试程序运行时间'''
start = clock()
finish = clock()
# print((finish-start)/1000000)

'''计算F1值-微平均'''
def F_score(test_data):
    test_data.columns = ['0', '1']
    list_one = list(test_data['0'])
    list_two = list(test_data['1'])
    print(list_one)
    print(list_two)
    sum_f = 0
    for index in range(test_data.shape[0]):
        word_one = set(list_one[index].split(','))
        word_two = list_two[index].split(',')
        p = 0; r = 0; sum = 0
        for string in word_two:
            if string in word_one:
                sum += 1
        p = sum / len(word_one)
        r = sum / len(word_two)
        # print(sum, len(word_one), len(word_two))
        if p != 0 or r != 0:
            sum_f += (2 * p * r) / (p + r)
    print('F-score:', sum_f / len(test_data))

'''获取数据集'''
def read_dataset():
    '''测试数据集'''
    read_data = pd.read_csv(r'D:\dataSet\NLP\update_data\news_tags.csv', encoding='utf-8', low_memory=False)    # , encoding="gb2312"  , nrows=20
    read_data = read_data.iloc[:, [0, 1, 2]]
    read_data = read_data[read_data['tags'].notnull()]
    read_data = read_data.drop_duplicates(['content', 'title', 'tags'])
    # print(read_data.shape)

    read_data['content'] = [str(string) for string in read_data['content']]
    read_data['title'] = [str(string) for string in read_data['title']]
    read_data['content'] = [delete(string) for string in read_data['content']]

    # # 将tags放入自定义词典中去 #
    # key_words_list = list(read_data['tags'])
    # for string in key_words_list:
    #     string = string.split(',')
    #     for i in string:
    #         jieba.add_word(i)

    # 将title加入到content里面 #
    for index in range(40):
        read_data['content'] = read_data['content'] + read_data['title']

    # read_data['text_rank'] = [jieba.analyse.textrank(string, topK=3, withWeight=False) for string in read_data['content']]
    # read_data['key_words'] = [string[0]+','+string[1]+','+string[2] if len(string) == 3 else string for string in read_data['text_rank']]
    # read_data = read_data.iloc[:, [0, 1, 3]]

    read_data['textrank'] = [','.join(jieba.analyse.textrank(string, topK=20, withWeight=False)) for string in read_data['content']]
    read_data['tf-idf'] = [','.join(jieba.analyse.extract_tags(string, topK=20,  withWeight=False)) for string in read_data['content']]
    return read_data

# '''获取数据集测试'''
# read_data = read_dataset()
# print(read_data)
# '''保存文件'''
# read_data.to_csv('dataSet.csv', index=None, encoding='utf-8')
# print(read_data.iloc[:, 2:5])

def deal_data():
    '''打标'''
    data = pd.read_csv(r'dataSet.csv')  #  , nrows=20
    # 训练集#
    # print(data.shape[0])
    length = int(data.shape[0] / 2)
    data = data.iloc[0:length+1]  #训练集#
    # data = data.iloc[-length:]  #测试集#
    print(data)
    # print(data['tf-idf'])

    data = data.iloc[:, [2, 4]]
    List_one = list(data['tags'])
    List_two = list(data['tf-idf'])
    count = []
    number = []
    train_data = []
    for n in range(len(List_two)):
        List_one[n] = List_one[n].split(',')
        List_one[n] = set(List_one[n])

        List_two[n] = List_two[n].split(',')
        for index in List_two[n]:
            if index in List_one[n]:
                number.append(1)
                count.append(n+1)
            else:
                number.append(0)
                count.append(n+1)
            train_data.append(index)

    train_data = pd.DataFrame(train_data)
    count = pd.DataFrame(count)
    number = pd.DataFrame(number)
    train_data.columns = ['word']
    number.columns = ['label']
    count.columns = ['count']
    count['word'] = train_data['word']
    count['label'] = number['label']  #训练集用这条#

    # count.to_csv('train_data.csv', index=None, encoding='utf-8')  # 保存训练集 #
    # count.to_csv('all_data.csv', index=None, encoding='utf-8') # 保存所有候选关键词，为了提取所有词性 #
    # number.to_csv('label.csv', index=None, encoding='utf-8')
    # count.to_csv('test_data.csv', index=None, encding='utf-8')  # 保存测试集 #
    # print(train_data)
    print(count)
    # print(number)

    # 查看正负样本比例 #
    # print(number[number['label'] == 1].count())
    # print(number[number['label'] == 0].count())
    # print(number)

'''打标测试'''
# deal_data()

def get_feature(train_data):
    '''char.csv ： 词性集合
        train_data.csv : 训练集候选词
        label_data : 训练集标签
    '''
    train_data['characteristic'] = [pseg.cut(string) for string in train_data['word']]
    char = []
    character = list(train_data['characteristic'])
    for string in character:
        for word, flag in string:
            break
        char.append(flag)
        # print(word, flag)

    # # 提取词性集合时使用 #
    # char = set(char)
    # char = list(char)
    # char = pd.DataFrame(char)
    # char.columns = ['char']
    # char.to_csv('char.csv', index=None, encoding='utf-8')
    # print(char)

    char = pd.DataFrame(char)
    char.columns = ['char']

    words = pd.read_csv(r'char.csv')
    list_set = list(words['char'])

    train_data['characteristic'] = [string for string in char['char']]
    # print(train_data)
    for index in list_set:
        train_data[index] = [1 if index_y == index else 0 for index_y in train_data['characteristic']]

    train_data['length'] = [len(string) for string in train_data['word']]
    # print(train_data)
    return train_data

def train_module():

    '''训练集提取特征'''
    # train_data = pd.read_csv(r'all_data.csv', encoding='utf-8')  # 获取词性集合时使用 #
    train_data = pd.read_csv(r'train_data.csv', encoding='utf-8')
    train_data = train_data[train_data['word'].notnull()]
    # print(train_data)
    train_data = get_feature(train_data)

    train_feature = train_data.iloc[:, -53: ]  #训练特征#
    label_data = train_data.iloc[:, [2]]  #训练标签#
    # print(train_feature)
    # print(label_data)

    '''测试集提取特征'''
    test_data = pd.read_csv(r'test_data.csv', encoding='utf-8')
    test_data = test_data[test_data['word'].notnull()]
    # print(test_data)
    test_data = get_feature(test_data)
    test_feature = test_data.iloc[:, -53: ]
    # print(test_feature)

    module = xgb.XGBClassifier(
        learning_rate=0.2,
        n_estimators=220,
        max_depth=9,
        gamma=0.1,
        subsample=0.7,
        objective='binary:logistic',
        nthread=5,
        scale_pos_weight=1,
        seed=27
    )

    module.fit(train_feature.values, label_data.values)
    proba = module.predict_proba(test_feature.values)[:, 1]
    proba = pd.DataFrame(proba)
    proba.columns = ['probability']
    print(proba)
    test_data = test_data.iloc[:, [0, 1, 3]]
    test_data['probability'] = [index for index in proba['probability']]
    test_data.to_csv('result.csv', index=None, encoding='utf-8')
    print(test_data)

# train_module()

result = pd.read_csv('result.csv', encoding='utf-8')
print(result.sort_values('probability'))

# F_score(read_data.iloc[:, [2, 3]])
# F_score(read_data.iloc[:, [2, 4]])
print('Finished!')

=
