#coding=utf-8

from time import clock
import pandas as pd
import jieba.analyse

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
# read_data = pd.read_csv(r'D:\dataSet\NLP\news_50000.csv', encoding='utf-8', low_memory=False)    # , encoding="gb2312"  , nrows=10
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
'''测试数据集'''
read_data = pd.read_csv(r'D:\dataSet\NLP\news_tags.csv', encoding='utf-8', low_memory=False, nrows=10)    # , encoding="gb2312"
# read_data = read_data.iloc[:, [0, 2]]
# print(read_data)

read_data['content'] = [str(string) for string in read_data['content']]
read_data['content'] = [delete(string) for string in read_data['content']]

# read_data['text_rank'] = [jieba.analyse.textrank(string, topK=3, withWeight=False) for string in read_data['content']]
# read_data['key_words'] = [string[0]+','+string[1]+','+string[2] if len(string) == 3 else string for string in read_data['text_rank']]
# read_data = read_data.iloc[:, [0, 1, 3]]

read_data['textrank'] = [','.join(jieba.analyse.textrank(string, topK=3, withWeight=False)) for string in read_data['content']]
read_data['tf-idf'] = [','.join(jieba.analyse.extract_tags(string, topK=3,  withWeight=False)) for string in read_data['content']]

read_data.to_csv('second_split.csv', index=None, encoding='utf-8')
print(read_data)

finish = clock()
print((finish-start)/1000000)
print('Finished!')
