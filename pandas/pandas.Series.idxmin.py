"""
Function : Return the row label of the minimum value.
"""

>>> import pandas as pd
>>>
>>> s = pd.Series(data=[1, None, 4, 1], index=['A', 'B', 'C', 'D'])
>>> s
A    1.0
B    NaN
C    4.0
D    1.0
dtype: float64
>>> s.idxmin()
'A'
>>> s.idxmax()
'C'
>>> s.idxmin(skipna=False)
nan
>>> s.idxmin(skipna=True)
'A'
>>>
