Examples

>>> df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1],
...                    [np.nan, np.nan, np.nan, 5]],
...                   columns=list('ABCD'))
>>> df
     A    B   C  D
0  NaN  2.0 NaN  0
1  3.0  4.0 NaN  1
2  NaN  NaN NaN  5
Drop the columns where all elements are nan:

>>> df.dropna(axis=1, how='all')
     A    B  D
0  NaN  2.0  0
1  3.0  4.0  1
2  NaN  NaN  5
Drop the columns where any of the elements is nan

>>> df.dropna(axis=1, how='any')
   D
0  0
1  1
2  5
Drop the rows where all of the elements are nan (there is no row to drop, so df stays the same):

>>> df.dropna(axis=0, how='all')
     A    B   C  D
0  NaN  2.0 NaN  0
1  3.0  4.0 NaN  1
2  NaN  NaN NaN  5
Keep only the rows with at least 2 non-na values:

>>> df.dropna(thresh=2)
     A    B   C  D
0  NaN  2.0 NaN  0
1  3.0  4.0 NaN  1

# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html #
