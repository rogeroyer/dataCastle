"""
Function:Convert categorical variable into dummy/indicator variables
prefix: string, list of strings, or dict of strings, default None
String to append DataFrame column names. Pass a list with length equal to the number of columns when calling get_dummies 
on a DataFrame. Alternatively, prefix can be a dictionary mapping column names to prefixes.
"""

>>> import pandas as pd
>>> list('abcd')
['a', 'b', 'c', 'd']
>>> s = pd.Series(list('abcd'))
>>> s
0    a
1    b
2    c
3    d
dtype: object
>>> pd.get_dummies(s)
   a  b  c  d
0  1  0  0  0
1  0  1  0  0
2  0  0  1  0
3  0  0  0  1
>>> import numpy as np
>>> s1 = ['a', 'b', np.nan]
>>> pd.get_dummies(s1)
   a  b
0  1  0
1  0  1
2  0  0
>>> pd.get_dummies(s1, dummy_na=True)
   a  b  NaN
0  1  0    0
1  0  1    0
2  0  0    1
>>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],'C':[1, 2, 3]})
>>> pd.get_dummies(df, prefix=['col1', 'col2'])
   C  col1_a  col1_b  col2_a  col2_b  col2_c
0  1       1       0       0       1       0
1  2       0       1       1       0       0
2  3       1       0       0       0       1
