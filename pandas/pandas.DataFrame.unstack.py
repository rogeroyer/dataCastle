>>> import pandas as pd
>>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'), ('two', 'a'), ('two', 'b')])
>>> s = pd.Series(np.arange(1.0, 5.0), index=index)
>>> s
one  a    1.0
     b    2.0
two  a    3.0
     b    4.0
dtype: float64
>>> s.unstack(level=-1)
       a    b
one  1.0  2.0
two  3.0  4.0
>>> s.unstack(level=0)
   one  two
a  1.0  3.0
b  2.0  4.0
>>>


'''example two'''
user_register = user_register.loc[0:100]
print(user_register)
print(user_register.groupby(by=['register_day', 'register_type'])['device_type'].mean())
print(user_register.groupby(by=['register_day', 'register_type'])['device_type'].max().unstack(level=-1))

output:
     user_id  register_day  register_type  device_type
0     167777             1              4          270
1     886972             1              0            5
2     921231             1              0            0
3     904908             1              1           49
4     460291             2              0           72
5    1096316             2              1         4912
6     641814             2              0           11
7     816839             2              1         1454
8     914942             3              0           67
9     450362             3              1            6
10    858989             3              0          136
11    754467             3              0           10
12   1291734             3              0           34
13    658865             3              0          420
14    655290             3              2           27
15     99574             4              2            0
16    800682             4              0           11
17   1308407             4              0           99
18    652726             4              0           51
19    200843             4              0           44
20    545515             5              1           72
21    242829             5              0          323
22   1021890             5              1          316
23    153141             5              0          580
24    501631             6              2          978
25    215787             6              0            3
26    269765             6              1           36
27    352546             6              1           95
28     98476             6              1           12
29   1071583             6              2           56
..       ...           ...            ...          ...
71    319366            11              1          323
72   1290640            12              0           76
73    279586            12              1         1719
74   1366429            12              1           63
75    271997            12              0            2
76    356112            12              2           13
77    231150            12              1           80
78     64843            12              0          372
79    316686            12              2          143
80    162427            12              0          173
81   1313891            12              1           11
82    924179            12              4            8
83   1370738            13              0           18
84    289642            13              5         2997
85   1143518            13              2           40
86    979146            13              1          116
87   1159043            13              2            2
88   1000296            13              0         1875
89    854604            13              1          294
90    742580            13              6         3004
91    322882            13              1           40
92    496410            13              1           51
93    253493            13              0            0
94    105807            13              0          263
95   1220321            13              1           25
96     66283            13              0           66
97    206211            13              0           11
98    832364            13              0            4
99    319137            13              0          104
100   111543            13              0         1031

[101 rows x 4 columns]
register_day  register_type
1             0                   2.500000
              1                  49.000000
              4                 270.000000
2             0                  41.500000
              1                3183.000000
3             0                 133.400000
              1                   6.000000
              2                  27.000000
4             0                  51.250000
              2                   0.000000
5             0                 451.500000
              1                 194.000000
6             0                   1.500000
              1                 111.400000
              2                 348.333333
7             0                  50.000000
              1                 210.750000
              3                  34.000000
              4                 520.000000
8             0                  26.000000
              1                  64.000000
              5                 133.500000
9             0                  97.000000
              1                  32.500000
              2                   7.000000
10            0                 426.666667
              1                 139.500000
              2                   2.000000
11            0                 422.000000
              1                 129.250000
12            0                 155.750000
              1                 468.250000
              2                  78.000000
              4                   8.000000
13            0                 374.666667
              1                 105.200000
              2                  21.000000
              5                2997.000000
              6                3004.000000
Name: device_type, dtype: float64
register_type       0       1      2     3      4       5       6
register_day                                                     
1                 5.0    49.0    NaN   NaN  270.0     NaN     NaN
2                72.0  4912.0    NaN   NaN    NaN     NaN     NaN
3               420.0     6.0   27.0   NaN    NaN     NaN     NaN
4                99.0     NaN    0.0   NaN    NaN     NaN     NaN
5               580.0   316.0    NaN   NaN    NaN     NaN     NaN
6                 3.0   398.0  978.0   NaN    NaN     NaN     NaN
7               169.0   439.0    NaN  34.0  520.0     NaN     NaN
8                55.0    78.0    NaN   NaN    NaN   257.0     NaN
9                97.0    37.0    7.0   NaN    NaN     NaN     NaN
10             2426.0   237.0    2.0   NaN    NaN     NaN     NaN
11              466.0   323.0    NaN   NaN    NaN     NaN     NaN
12              372.0  1719.0  143.0   NaN    8.0     NaN     NaN
13             1875.0   294.0   40.0   NaN    NaN  2997.0  3004.0
