#coding=utf-8
'''Create by roger  2017/11/18 '''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor


'''店铺的广告信息'''
# t_ads = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_ads.csv', low_memory=False)
'''店铺的评论信息'''
# t_comment = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_comment.csv', low_memory=False)
'''店铺的订单信息'''
t_order = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_order.csv', low_memory=False)
'''店铺的商品信息'''
# t_product = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_product.csv', low_memory=False)
'''店铺每月末后90天的销售额'''
t_sales_sum = pd.read_csv(r't_sales_sum.csv', low_memory=False)

'''判断是否含有空值'''
def judge_isnull(data):
    col = list(data.columns)
    for co in col:
        print(data[data[co].isnull()])

'''获取数据集-训练集&测试集'''
def get_train_data(date_one, date_two):
    divided_data = t_order[(t_order['ord_dt'] >= date_one) & (t_order['ord_dt'] <= date_two)]
    divided_data = divided_data.sort_values(by='shop_id')
    return divided_data

'''获取预测值'''
def get_label_data(date):
    divided_data = t_sales_sum[t_sales_sum['dt'] == date]
    divided_data = divided_data.iloc[:, [2]]
    return divided_data

'''众数函数'''
def MostOccur(group):
    d = {}
    s = set()
    for x in group:
        if x in s:
            d[x] += 1;
        else:
            s.add(x)
            d[x] = 1
    for key in d:
        if d[key] == max(d.values()):
            return key

'''提取特征'''
def get_feature_data(dataset):
    # dataset['ord_dt'] = pd.to_datetime(dataset['ord_dt'])
    '''offer_cnt'''
    feature = pd.pivot_table(dataset, index='shop_id', values='offer_cnt', aggfunc='mean')
    feature['shop_id'] = feature.index
    feature.columns = ['feature_1', 'shop_id']
    feature_2 = pd.pivot_table(dataset, index='shop_id', values='offer_cnt', aggfunc='max')
    feature_3 = pd.pivot_table(dataset, index='shop_id', values='offer_cnt', aggfunc='var')
    feature_4 = pd.pivot_table(dataset, index='shop_id', values='offer_cnt', aggfunc=MostOccur)
    feature_5 = pd.pivot_table(dataset, index='shop_id', values='offer_cnt', aggfunc='sum')
    '''offer_amt'''
    feature_6 = pd.pivot_table(dataset, index='shop_id', values='offer_amt', aggfunc='mean')
    feature_7 = pd.pivot_table(dataset, index='shop_id', values='offer_amt', aggfunc='max')
    feature_8 = pd.pivot_table(dataset, index='shop_id', values='offer_amt', aggfunc='var')
    feature_9 = pd.pivot_table(dataset, index='shop_id', values='offer_amt', aggfunc=MostOccur)
    feature_10 = pd.pivot_table(dataset, index='shop_id', values='offer_amt', aggfunc='median')
    '''rtn_cnt'''
    feature_11 = pd.pivot_table(dataset, index='shop_id', values='rtn_cnt', aggfunc='sum')
    feature_12 = pd.pivot_table(dataset, index='shop_id', values='rtn_cnt', aggfunc='mean')
    feature_13 = pd.pivot_table(dataset, index='shop_id', values='rtn_cnt', aggfunc='max')
    '''rtn_amt'''
    feature_14 = pd.pivot_table(dataset, index='shop_id', values='rtn_amt', aggfunc='max')
    feature_15 = pd.pivot_table(dataset, index='shop_id', values='rtn_amt', aggfunc='sum')
    feature_16 = pd.pivot_table(dataset, index='shop_id', values='rtn_amt', aggfunc='mean')
    '''user_cnt'''
    feature_17 = pd.pivot_table(dataset, index='shop_id', values='user_cnt', aggfunc='mean')
    feature_18 = pd.pivot_table(dataset, index='shop_id', values='user_cnt', aggfunc='max')
    feature_19 = pd.pivot_table(dataset, index='shop_id', values='user_cnt', aggfunc='sum')
    '''sale_amt'''
    feature_20 = pd.pivot_table(dataset, index='shop_id', values='sale_amt', aggfunc='mean')
    feature_21 = pd.pivot_table(dataset, index='shop_id', values='sale_amt', aggfunc='median')
    feature_22 = pd.pivot_table(dataset, index='shop_id', values='sale_amt', aggfunc='var')
    '''ord_cnt'''
    # feature_23 = pd.pivot_table(dataset, index='shop_id', values='ord_cnt', aggfunc='var')
    # feature_24 = pd.pivot_table(dataset, index='shop_id', values='ord_cnt', aggfunc='var')
    # feature_25 = pd.pivot_table(dataset, index='shop_id', values='ord_cnt', aggfunc='var')

    feature['feature_2'] = feature_2['offer_cnt']
    feature['feature_3'] = feature_3['offer_cnt']
    feature['feature_4'] = feature_4['offer_cnt']
    feature['feature_5'] = feature_5['offer_cnt']

    feature['feature_6'] = feature_6['offer_amt']
    feature['feature_7'] = feature_7['offer_amt']
    feature['feature_8'] = feature_8['offer_amt']
    feature['feature_9'] = feature_9['offer_amt']
    feature['feature_10'] = feature_10['offer_amt']

    feature['feature_11'] = feature_11['rtn_cnt']
    feature['feature_12'] = feature_12['rtn_cnt']
    feature['feature_13'] = feature_13['rtn_cnt']

    feature['feature_14'] = feature_14['rtn_amt']
    feature['feature_15'] = feature_15['rtn_amt']
    feature['feature_16'] = feature_16['rtn_amt']

    feature['feature_17'] = feature_17['user_cnt']
    feature['feature_18'] = feature_18['user_cnt']
    feature['feature_19'] = feature_19['user_cnt']

    feature['feature_20'] = feature_20['sale_amt']
    feature['feature_21'] = feature_21['sale_amt']
    feature['feature_22'] = feature_22['sale_amt']

    feature = feature.drop('shop_id', axis=1)
    return feature



def get_sale_amount():
    '''2016-11-01 -- 2017-01-31'''
    train_data = get_train_data("2016-11-01", "2017-01-31")
    test_data = get_train_data("2017-02-01", "2017-04-30")
    label_data = get_label_data("2017-01-31")
    # print(train_data)
    # print(test_data)
    label_data.columns = [0]
    train_feature_data = get_feature_data(train_data)
    test_feature_data = get_feature_data(test_data)
    '''2016-10-01 -- 2016-12-31'''
    train_data_one = get_train_data("2016-10-01", "2016-12-31")
    train_feature_data_one = get_feature_data(train_data_one)
    label_data_one = get_label_data("2016-12-31")
    train_feature_data = train_feature_data.append(train_feature_data_one)
    label_data_one.columns = [0]
    label_data = label_data.append(label_data_one)
    '''2016-09-01 -- 2016-11-30'''
    train_data_two = get_train_data("2016-09-01", "2016-11-30")
    train_feature_data_two = get_feature_data(train_data_two)
    label_data_two = get_label_data("2016-11-30")
    train_feature_data = train_feature_data.append(train_feature_data_two)
    label_data_two.columns = [0]  # label_data要更改列名 #
    label_data = label_data.append(label_data_two)

    # axis_y = train_feature_data.shape()
    # print(axis_y)

    train_feature_data.columns = [index for index in range(22)]               # 加了特征之后要把这个值改为特征的维数 #
    test_feature_data.columns = [index for index in range(22)]                # 加了特征之后要把这个值改为特征的维数 #
    train_feature_data = train_feature_data.fillna(0)
    test_feature_data = test_feature_data.fillna(0)
    # print(train_feature_data)
    # print(test_feature_data)
    # print(label_data)

    '''验证是否含有空值'''
    # judge_isnull(train_feature_data)
    # judge_isnull(test_feature_data)

    # module = LogisticRegression()
    module = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
    module.fit(train_feature_data, label_data)
    result = module.predict(test_feature_data)
    result = pd.DataFrame(result)
    result['shop_id'] = [index for index in range(1, 3001)]
    result['values'] = result[0]
    result = result.drop(0, axis=1)

    '''验证'''
    yz_data = get_train_data("2016-08-01", "2016-10-31")
    yz_label_data = get_label_data("2016-10-31")
    # print('yz_label_data\n', yz_label_data)
    # yz_label_data.columns = [0]  # columns=['sale_amt_3m'] #
    yz_feature_data = get_feature_data(yz_data)
    yz_feature_data.columns = [index for index in range(22)]
    # print(yz_feature_data)
    yz_feature_data = yz_feature_data.fillna(0)
    yz_result = module.predict(yz_feature_data)
    yz_result = pd.DataFrame(yz_result)
    yz_result.columns = ['result']
    # print(yz_result)
    # print(yz_label_data)
    yz_result['sale_amt_3m'] = [index for index in yz_label_data['sale_amt_3m']]
    # print(yz_result)
    yz_button = sum(yz_result['sale_amt_3m'])
    # print(yz_button)
    yz_result['cha'] = abs(yz_result['sale_amt_3m'] - yz_result['result'])
    yz_top = sum(yz_result['cha'])
    # print(yz_top)
    result_rate = yz_top / yz_button
    print(result_rate)
    # print(yz_result)
    '''验证结束'''

    return result
    # from sklearn import tree
    # clf = tree.DecisionTreeRegressor()
    # clf = clf.fit(X, y)
    # clf.predict()

    # from sklearn import svm
    # >> > X = [[0, 0], [2, 2]]
    # >> > y = [0.5, 2.5]
    # clf = svm.SVR()
    # clf.fit(X, y)
    # clf.predict([[1, 1]])

values = get_sale_amount()
print(values)
values.to_csv('result.csv', index=None, header=None)
print('Finished!')

result = 0.488792
