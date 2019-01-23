"""
Function:Return boolean DataFrame showing whether each element in the DataFrame is contained in values.
"""

>>> import pandas as pd
>>> df = pd.DataFrame({'A':[1, 2, 3], 'B':['a', 'b', 'c']})
>>> df
   A  B
0  1  a
1  2  b
2  3  c
>>> df.isin([1, 3, 12, 'a'])
       A      B
0   True   True
1  False  False
2   True  False
>>> df.isin({'A':[1, 3], 'B':[4, 7, 12]})
       A      B
0   True  False
1  False  False
2   True  False

>>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
>>> other = pd.DataFrame({'A': [1, 3, 3, 2], 'B': ['e', 'f', 'f', 'e']})
>>> df.isin(other)
       A      B
0   True  False
1  False  False
2   True   True

>>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
>>> other = pd.DataFrame({'A': [3, 3, 3, 2], 'B': ['e', 'f', 'b', 'e']})
>>> df.isin(other)
       A      B
0  False  False
1  False  False
2   True  False
