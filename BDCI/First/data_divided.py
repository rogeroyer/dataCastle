#coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt


'''读取数据集'''
# # 用户行为表 #
# read_data_one = pd.read_csv(r'D:\dataSet\aliyun_ccf\train_data-ccf_first_round_user_shop_behavior.csv')
# # AB榜测试集 #
read_data_two = pd.read_csv(r'D:\dataSet\aliyun_ccf\ABtest-evaluation_public.csv')
# # 商场和店铺信息 #
read_data_three = pd.read_csv(r'D:\dataSet\aliyun_ccf\train_data-ccf_first_round_shop_info.csv')
# print (read_data_two['time_stamp'].max())


'''划分测试集并筛选'''
read_data_two = read_data_two.merge(read_data_three, on='mall_id', how='left')
# read_data_two = read_data_two[(abs(read_data_two['longitude_y'] - read_data_two['longitude_x']) < 0.0004) & (abs(read_data_two['latitude_y'] - read_data_two['latitude_x']) < 0.0004)]  # 阈值:0.0004 #
read_data_two.to_csv('test_data.csv', index=None)
print(read_data_two)
print(read_data_two.drop_duplicates(['row_id']))


'''数据预处理'''
# # 处理时间戳 #
# read_data_one['date'] = [index.split(' ')[0] for index in read_data_one['time_stamp']]
# read_data_one['date'] = [int(index.replace('-', '')) for index in read_data_one['date']]
# read_data_one['date'] = [str(index) for index in read_data_one['date']]
# print read_data_one['date'].max()
# print read_data_one['date'].min()
# # 统计每天有多少条记录 #
# feature = pd.pivot_table(read_data_one, index='date', values='user_id', aggfunc='count')
# print feature
# # 画折线图 #
# # plt.plot(feature)
# # plt.show()
# # 对时间的处理 #
# read_data_one['time'] = [index.split(' ')[1] for index in read_data_one['time_stamp']]
# read_data_one['hour'] = [index.split(':')[0] for index in read_data_one['time']]
# read_data_one['minute'] = [index.split(':')[1] for index in read_data_one['time']]


# # 分析每个用户在8月份有多少条记录 #
# feature = pd.pivot_table(read_data_one, index='user_id', values='time_stamp', aggfunc='count')
# # feature.index = range(len(feature))
# # print feature
# feature['index'] = range(len(feature))
# X = list(feature['index'])
# Y = list(feature['time_stamp'])
# plt.plot(X, Y, color='red')
# plt.show()


# # 每个商场的店铺数量 #
# feature = pd.pivot_table(read_data_three, index='mall_id', values='shop_id', aggfunc='count')
# feature.index = range(len(feature))
# X = list(feature.index)
# Y = list(feature['shop_id'])
# plt.plot(X, Y)
# plt.show()

# # 计算wifi个数 #
# read_data_one['wifi_count'] = [len(index.split(';')) for index in read_data_one['wifi_infos']]
# print read_data_one
# X = read_data_one.index
# Y = list(read_data_one['wifi_count'])
# plt.scatter(X, Y)
# plt.show()

'''划分验证集和训练集'''
# trian_data_one = read_data_one[(read_data_one['date'] >= 20170801) & (read_data_one['date'] < 20170816)]
# trian_data_two = read_data_one[(read_data_one['date'] >= 20170816) & (read_data_one['date'] <= 20170831)]
# trian_data_one.to_csv('train_data_one.csv', index=None)
# trian_data_two.to_csv('train_data_two.csv', index=None)

# train_data_one = pd.read_csv(r'train_data_one.csv')  # , nrows=50000 # # 验证集 #
# train_data_one = train_data_one.merge(read_data_three.iloc[:, [0, 5]], on='shop_id', how='left')
# train_data_one = train_data_one.merge(read_data_three, on='mall_id', how='left')
# # print train_data_one
#
# train_data_one = train_data_one.iloc[:, [1, 8, 3, 10, 4, 11]]
# print train_data_one[train_data_one['shop_id_x'] == train_data_one['shop_id_y']]
# train_data_two = pd.read_csv(r'train_data_two.csv') # 训练集 #
# train_data_one = train_data_one.drop_duplicates(['user_id']).iloc[:, [0, 1]]
# train_data_two = train_data_two.drop_duplicates(['user_id']).iloc[:, [0, 1]]
# print len(train_data_one)
# print len(train_data_two)
# train_data_one = train_data_one.merge(train_data_two, on='user_id', how='left')
# print train_data_one[train_data_one['shop_id_y'].notnull()]

''' 筛选负样本 '''
# read_data_four = read_data_three.iloc[:, [0, 4, 5]]  # 验证集 #
# train_data_one = train_data_one.merge(read_data_four, on='shop_id', how='left')
# train_data_one = train_data_one.merge(read_data_three, on='mall_id', how='left')
# train_data_one = train_data_one[abs(train_data_one['price_y'] - train_data_one['price_x']) < 3]
# train_data_one.to_csv('verify_data.csv', index=None)
# print(train_data_one)
# read_data_four = read_data_three.iloc[:, [0, 4, 5]] #训练集#
# train_data_two = train_data_two.merge(read_data_four, on='shop_id', how='left')
# train_data_two = train_data_two.merge(read_data_three, on='mall_id', how='left')
# train_data_two = train_data_two[abs(train_data_two['price_y'] - train_data_two['price_x']) < 3]
# train_data_two.to_csv('train_data.csv', index=None)
# print(train_data_two)
# [105749945 rows x 12 columns] #

# read_data_one['number'] = [len(index.split(';')) for index in read_data_one['wifi_infos']]
# print(read_data_one['number'].mean())

# feature = pd.pivot_table(read_data_three, index='mall_id', values='price', aggfunc='max')
# feature2 = pd.pivot_table(read_data_three, index='mall_id', values='price', aggfunc='min')
# feature['two'] = feature2['price']
# print(feature)
'''计算Seris长度'''
def call_count(group):
    return len(list(group))

#########################################################################################
#########################################################################################
''''打标'''
# train_data = pd.read_csv(r'train_data.csv')  #, nrows=100
# train_data['_index'] = train_data.index
# positive_sample = train_data[train_data['shop_id_x'] == train_data['shop_id_y']].iloc[:, [-1]]
# positive_sample['label'] = [1 for index in range(len(positive_sample))]
# train_data = train_data.merge(positive_sample, on='_index', how='left')
# train_data.label = train_data.label.fillna(0)
# label_data = train_data.iloc[:, [-1]]
# # label_data.to_csv('label_data.csv', index=None)
# # print(label_data)

# 616675  521340 #
# train_feature = train_data.iloc[:, [0]]
# train_feature.to_csv('train_feature.csv', index=None)
# print(train_feature)

'''透视表提取特征函数'''
def pivot_feature(table,_aggfunc,_index='user_id',_values=None):
    dataset = pd.pivot_table(table, index=_index, values=_values, aggfunc=_aggfunc)
    dataset[_index] = dataset.index
    train_feature = pd.read_csv(r'train_feature.csv')
    train_feature = train_feature.merge(dataset, on=_index, how='left')
    # train_feature.to_csv('train_feature.csv', index=None)
    print(train_feature)

'''just for test'''
# pivot_feature(train_data, 'mean', 'user_id', 'price_y')

# train_data = pd.read_csv(r'train_data.csv')  #, nrows=100

# c_49,c_19,c_36,c_6,c_4,c_11,c_50,c_55,c_1,c_53...c_8 \
    #'''提取category_id相关特征'''#
def category_feature(data):
    train_data = pd.read_csv(r'train_data.csv')  # 提取测试集特征需要读取 #
    list_set = list(train_data['category_id'].drop_duplicates())
    feature = pd.read_csv(r'test_feature.csv')          #数据集不同表名也不同#
    for index in list_set:
        feature[index] = [1 if index_y == index else 0 for index_y in data['category_id']]
    feature['price'] = data['price']      #数据集不同列名也不同 ## 训练集是train_data## 测试集是test_data#
    feature.to_csv('test_feature.csv', index=None)   #数据集不同表名也不同#
    print(feature)

# test_data = pd.read_csv(r'test_data.csv')
# test_feature = test_data.iloc[:, [0, 7]]
# # print(test_feature)
# test_feature.to_csv('test_feature.csv', index=None)
# print(test_feature)
# # category_feature(test_data)

