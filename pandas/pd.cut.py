>>> import numpy as np
>>> import pandas as pd
>>> from pandas import Series, DataFrame

>>> score_list = np.random.randint(25, 100, size=20)
>>> score_list
array([52, 35, 83, 35, 27, 45, 35, 83, 92, 35, 64, 61, 46, 87, 56, 57, 87,
       42, 37, 65])
>>> bins = [0, 59, 70, 80, 100]

>>> pd.cut(score_list, bins)
[(0, 59], (0, 59], (80, 100], (0, 59], (0, 59], ..., (0, 59], (80, 100], (0, 59], (0, 59], (59, 70]]
Length: 20
Categories (4, interval[int64]): [(0, 59] < (59, 70] < (70, 80] < (80, 100]]

>>> score_cut = pd.cut(score_list, bins)
>>> score_cut
[(0, 59], (0, 59], (80, 100], (0, 59], (0, 59], ..., (0, 59], (80, 100], (0, 59], (0, 59], (59, 70]]
Length: 20
Categories (4, interval[int64]): [(0, 59] < (59, 70] < (70, 80] < (80, 100]]

>>> pd.value_counts(score_cut)
(0, 59]      12
(80, 100]     5
(59, 70]      3
(70, 80]      0
dtype: int64
>>> df = DataFrame()
>>> df['score'] = score_list
>>> df.head()
   score
0     52
1     35
2     83
3     35
4     27

# 这里的pd.util.testing.rands(3) for i in range(20)可以生成20个随机3位字符串。
>>> df['student'] = [pd.util.testing.rands(3) for i in range(20)]
>>> df.head()
   score student
0     52     QHW
1     35     hyt
2     83     DZE
3     35     Pl7
4     27     eHE
>>> df['categories'] = pd.cut(df['score'], bins)
>>> df.head()
   score student categories
0     52     QHW    (0, 59]
1     35     hyt    (0, 59]
2     83     DZE  (80, 100]
3     35     Pl7    (0, 59]
4     27     eHE    (0, 59]

# 这样子可读性不好，可以指定label参数为每个区间赋一个标签：
>>> df['categories'] = pd.cut(df['score'], bins, labels=['low', 'ok', 'good', 'great'])
>>> df.head()
   score student categories
0     52     QHW        low
1     35     hyt        low
2     83     DZE      great
3     35     Pl7        low
4     27     eHE        low
>>>
