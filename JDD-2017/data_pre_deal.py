#coding=utf-8
'''Create by roger  2017/11/18 '''
import pandas as pd

'''店铺的广告信息'''
# t_ads = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_ads.csv', low_memory=False)  ######################
# create_dt     charge    consume  shop_id #
# [226128 rows x 4 columns] #
'''店铺的评论信息'''
t_comment = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_comment.csv', low_memory=False)  ##############
# create_dt  bad_num  cmmt_num  dis_num  good_num  mid_num  shop_id #
# [636206 rows x 7 columns] #
'''店铺的订单信息'''
# t_order = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_order.csv', low_memory=False)  ##################
# ord_dt  sale_amt  offer_amt  offer_cnt  shop_id  rtn_cnt   rtn_amt  ord_cnt  pid  user_cnt #
# [12098397 rows x 10 columns] #
'''店铺的商品信息'''
# t_product = pd.read_csv(r'D:\dataSet\JDD\Sales_Forecast_Qualification\t_product.csv', low_memory=False)  ##############
# on_dt off_dt  brand  cate  shop_id       pid #
# [11398239 rows x 6 columns] #
'''店铺每月末后90天的销售额'''
t_sales_sum = pd.read_csv(r't_sales_sum.csv', low_memory=False)  ##########
# dt  shop_id  sale_amt_3m #
# [24030 rows x 3 columns] #

# print('t_ads\n', t_ads)
# print('t_comment\n', t_comment)
# print('t_order\n', t_order)
# print('t_product\n', t_product)
# print('t_sales_sum\n', t_sales_sum)

'''判断是否含有空值'''
def judge_isnull(data):
    col = list(data.columns)
    for co in col:
        print(data[data[co].isnull()])


'''t_sales_sum'''
# print(t_sales_sum[(t_sales_sum['dt'] >= '2017-01-01') & (t_sales_sum['dt'] <= '2017-01-31')])
# print(t_sales_sum['dt'].max())
# print(t_sales_sum['dt'].min())
'''判断节假日是否影响销售额'''
print(t_sales_sum)

'''t_product'''
# dataSet = t_product[(t_product['on_dt'] >= '2011-04-26') & (t_product['on_dt'] < '2016-06-01')]
# dataSet = t_product.copy()
# dataSet = dataSet.sort_values(by=['shop_id', 'on_dt'], ascending=True)
# dataSet = dataSet.drop_duplicates(['shop_id'])
# print(dataSet)

# dataSet = t_product[(t_product['on_dt'] >= '2017-03-01') & (t_product['on_dt'] < '2017-04-30')]
# dataSet = dataSet.sort_values(by=['shop_id', 'on_dt'], ascending=True)
# print(dataSet)
# dataSet = dataSet.drop_duplicates(['shop_id'])
# print(dataSet)

'''t_order'''
# print(t_order[t_order['ord_cnt'] != t_order['user_cnt']])
# print(t_order['ord_dt'].max())
# print(t_order['ord_dt'].min())
# dataSet = t_order.copy()
# dataSet = dataSet.sort_values(by=['shop_id', 'ord_dt'], ascending=True)
# dataSet = dataSet.drop_duplicates(['shop_id'])
# print(dataSet)

# dataSet = t_order[(t_order['ord_dt'] >= '2017-04-01') & (t_order['ord_dt'] < '2017-04-30')]
# dataSet = dataSet.sort_values(by=['shop_id', 'ord_dt'], ascending=True)
# print(dataSet)
# dataSet = dataSet.drop_duplicates(['shop_id'])
# print(dataSet)


'''t_comment'''
# dataSet = t_comment.copy()
# dataSet = dataSet.sort_values(by=['shop_id', 'create_dt'], ascending=True)
# dataSet = dataSet.drop_duplicates(['shop_id'])
# print(dataSet)
# dataSet = t_comment[(t_comment['create_dt'] >= '2017-04-01') & (t_comment['create_dt'] < '2017-04-30')]
# dataSet = dataSet.sort_values(by=['shop_id', 'create_dt'], ascending=True)
# print(dataSet)
# dataSet = dataSet.drop_duplicates(['shop_id'])
# print(dataSet)

'''t_ads'''
# print(t_ads)
# print(t_ads['create_dt'].max())
# print(t_ads['create_dt'].min())
# dataSet = t_ads.copy()
# dataSet = dataSet.sort_values(by=['shop_id', 'create_dt'], ascending=True)
# dataSet = dataSet.drop_duplicates(['shop_id'])
# print(dataSet)
