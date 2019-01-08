"""
Return Series with number of distinct observations over requested axis.
"""


Example one:
>>> df = pd.DataFrame({'A':[1, 2, 3, 4], 'B':[1, 1, 1, 1], 'C':[5, 6, 7, 8]})
>>> df
   A  B  C
0  1  1  5
1  2  1  6
2  3  1  7
3  4  1  8
>>> df.nunique()
A    4
B    1
C    4
dtype: int64
>>> df.nunique(axis=1)
0    2
1    3
2    3
3    3
dtype: int64


Example two:
>>> df = pd.DataFrame({'A':[1, 2, 3], 'B':[1, 1, 1]})
>>> df
   A  B
0  1  1
1  2  1
2  3  1
>>> df.nunique()
A    3
B    1
dtype: int64
>>> df.nunique(axis=1)
0    1
1    2
2    2
dtype: int64
