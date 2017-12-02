## [官方网站](http://jddjr.jd.com/item/2)
> shop_id: 1-3000
> code:
> print('t_ads\n', t_ads.drop_duplicates(['shop_id']).sort_values(by='shop_id', ascending=False))
> print('t_comment\n', t_comment.drop_duplicates(['shop_id']).sort_values(by='shop_id', ascending=False))
> print('t_order\n', t_order.drop_duplicates(['shop_id']).sort_values(by='shop_id', ascending=False))
> print('t_product\n', t_product.drop_duplicates(['shop_id']).sort_values(by='shop_id', ascending=False))
> print('t_sales_sum\n', t_sales_sum.drop_duplicates(['shop_id']).sort_values(by='shop_id', ascending=False))

### t_sales_sum
> t_sales_sum 每个shop_id的起始时间
> 共8个月的数据
> 2016-06-30
> 2016-07-31
> 2016-08-31
> 2016-09-30
> 2016-10-31
> 2016-11-30
> 2016-12-31
> 2017-01-31
> 
> t_sales_sum 不止对应8个月的shop_id
> 2129      9
> 2205     37

- t_sales_sum删除重复行

```python
print(t_sales_sum.drop_duplicates(['shop_id', 'dt']))
# 将t_sales_sum按shop_id和dt降序排列
t_sales_sum = t_sales_sum.sort_values(by=['shop_id', 'dt'], ascending=True)
```

### t_product

> 最近时间：2017-05-01
> 最远时间：2011-04-26
> 下架商品信息：
> print(t_product[t_product['off_dt'].notnull()])
> [100682 rows x 6 columns]
> 只有t_product表里包含空值，其它表都不包含
> 
> 2016-06-01 - 2016-06-30: [407168 rows x 6 columns]
> 2016-07-01 - 2016-07-31: [632456 rows x 6 columns]
> 2016-08-01 - 2016-08-31: [824115 rows x 6 columns]
> 2016-09-01 - 2016-09-30: [914965 rows x 6 columns]
> 2016-10-01 - 2016-10-31: [780746 rows x 6 columns]
> 2016-11-01 - 2016-11-30: [569374 rows x 6 columns]
> 2016-12-01 - 2016-12-31: [565083 rows x 6 columns]
> 2017-01-01 - 2017-01-31: [276483 rows x 6 columns]
> shop_id ; [2886 rows x 6 columns]  即缺失一定的shop_id

### t_order 

> 最近时间: 2017-04-30
> 最远时间: 2016-08-03
> shop_id: 3000个

### t_comment 

> 最近时间: 2017-04-30
> 最远时间: 2016-08-04
> shop_id: 3000个

### t_ads 

> 最近时间: 2017-04-30
> 最远时间: 2016-08-04
> shop_id:[1951 rows x 4 columns]

- 粗糙结果

```python
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
# result = pd.pivot_table(t_sales_sum, index='shop_id', values='sale_amt_3m', aggfunc=MostOccur)  # MostOccur #
# result['shop_id'] = result.index
# result['amount'] = result['sale_amt_3m']
# result = result.drop('sale_amt_3m', axis=1)
# print(result)
# result.to_csv('sub_sample.csv', index=None, header=None)
```

> #################################################
>     train_data = get_train_data("2016-12-01", "2017-01-31")
>     test_data = get_train_data("2017-02-01", "2017-04-30")
>     label_data = get_label_data("2017-01-31")
> 
>     module = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
>     result = 0.63443
> #################################################
>     train_data = get_train_data("2016-11-01", "2017-01-31")
>     test_data = get_train_data("2017-01-01", "2017-04-30")
>     label_data = get_label_data("2017-01-31")
> 
>     module = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
>     result = 0.541434
> #################################################
