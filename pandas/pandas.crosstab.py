>>> import numpy as np
>>> import pandas as pd
>>>
>>> a = np.arange(10)
>>> b = np.arange(10,20)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
>>> date = pd.date_range('2016-8-26', periods = 10)
>>> date
DatetimeIndex(['2016-08-26', '2016-08-27', '2016-08-28', '2016-08-29',
               '2016-08-30', '2016-08-31', '2016-09-01', '2016-09-02',
               '2016-09-03', '2016-09-04'],
              dtype='datetime64[ns]', freq='D')
>>> df1 = pd.DataFrame({'date': date, 'SecName':'sec1', 'price': a})
>>> df1
  SecName       date  price
0    sec1 2016-08-26      0
1    sec1 2016-08-27      1
2    sec1 2016-08-28      2
3    sec1 2016-08-29      3
4    sec1 2016-08-30      4
5    sec1 2016-08-31      5
6    sec1 2016-09-01      6
7    sec1 2016-09-02      7
8    sec1 2016-09-03      8
9    sec1 2016-09-04      9
>>> df2 = pd.DataFrame({'date': date, 'SecName':'sec2', 'price': b})
>>> df2
  SecName       date  price
0    sec2 2016-08-26     10
1    sec2 2016-08-27     11
2    sec2 2016-08-28     12
3    sec2 2016-08-29     13
4    sec2 2016-08-30     14
5    sec2 2016-08-31     15
6    sec2 2016-09-01     16
7    sec2 2016-09-02     17
8    sec2 2016-09-03     18
9    sec2 2016-09-04     19
>>> df = pd.concat([df1,df2], axis = 0) #将两个数据框按行拼接
>>> df
  SecName       date  price
0    sec1 2016-08-26      0
1    sec1 2016-08-27      1
2    sec1 2016-08-28      2
3    sec1 2016-08-29      3
4    sec1 2016-08-30      4
5    sec1 2016-08-31      5
6    sec1 2016-09-01      6
7    sec1 2016-09-02      7
8    sec1 2016-09-03      8
9    sec1 2016-09-04      9
0    sec2 2016-08-26     10
1    sec2 2016-08-27     11
2    sec2 2016-08-28     12
3    sec2 2016-08-29     13
4    sec2 2016-08-30     14
5    sec2 2016-08-31     15
6    sec2 2016-09-01     16
7    sec2 2016-09-02     17
8    sec2 2016-09-03     18
9    sec2 2016-09-04     19



>>> df = pd.crosstab(df['date'], df['SecName']） # 第一个参数是指定index，第二个参数是指定column
>>> df = pd.crosstab(df['date'], df['SecName'])
>>> df
SecName     sec1  sec2
date
2016-08-26     1     1
2016-08-27     1     1
2016-08-28     1     1
2016-08-29     1     1
2016-08-30     1     1
2016-08-31     1     1
2016-09-01     1     1
2016-09-02     1     1
2016-09-03     1     1
2016-09-04     1     1



>>> df = pd.concat([df1,df2], axis = 0) #将两个数据框按行拼接
>>> df = pd.crosstab(df['date'], df['SecName'],values=df['price'],aggfunc=sum)
>>> df
SecName     sec1  sec2
date
2016-08-26     0    10
2016-08-27     1    11
2016-08-28     2    12
2016-08-29     3    13
2016-08-30     4    14
2016-08-31     5    15
2016-09-01     6    16
2016-09-02     7    17
2016-09-03     8    18
2016-09-04     9    19

# https://blog.csdn.net/alanguoo/article/details/52330404 #
