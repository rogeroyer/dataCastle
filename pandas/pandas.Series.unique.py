"""
Function:Return unique values in the object. Uniques are returned in order of appearance, this does NOT sort. Hash table-based unique.
"""

Example:
>>> df = pd.DataFrame({'A':[1, 2, 3, 4], 'B':[1, 1, 1, 1], 'C':[5, 6, 7, 5]})
>>> df
   A  B  C
0  1  1  5
1  2  1  6
2  3  1  7
3  4  1  5
>>> pd.unique(df['C'])
array([5, 6, 7], dtype=int64)
>>> df['C'].unique()
array([5, 6, 7], dtype=int64)
