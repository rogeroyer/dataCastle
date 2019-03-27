"""
Function:Return a random sample of items from an axis of object.
"""

>>> s = pd.DataFrame({'A':[1,1,1,1,2,2,2,3,3,3,3], 'B':[1,0,1,1,0,0,1,1,0,0,1], 'C':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.34, 0.8, 0.9, 0.1]})
>>> s
    A  B     C
0   1  1  0.10
1   1  0  0.20
2   1  1  0.30
3   1  1  0.40
4   2  0  0.50
5   2  0  0.60
6   2  1  0.70
7   3  1  0.34
8   3  0  0.80
9   3  0  0.90
10  3  1  0.10
>>> s['C'].sample(n=3)
0    0.1
5    0.6
6    0.7
Name: C, dtype: float64
>>> s['C'].sample(n=6)
0     0.1
6     0.7
5     0.6
9     0.9
2     0.3
10    0.1
Name: C, dtype: float64
>>> s['C'].sample(n=6)
4    0.50
2    0.30
8    0.80
7    0.34
1    0.20
3    0.40
Name: C, dtype: float64
>>> s['C'].sample(n=6)
0    0.1
5    0.6
9    0.9
8    0.8
6    0.7
3    0.4
Name: C, dtype: float64
>>> s['C'].sample(n=6)
3    0.4
1    0.2
2    0.3
5    0.6
0    0.1
8    0.8
Name: C, dtype: float64
>>> s['C'].sample(n=6)
3    0.4
6    0.7
8    0.8
9    0.9
0    0.1
2    0.3
Name: C, dtype: float64
>>> s['C'].sample(n=6)
3     0.4
0     0.1
10    0.1
9     0.9
8     0.8
6     0.7
Name: C, dtype: float64
>>> s.sample(n=6)
    A  B    C
3   1  1  0.4
1   1  0  0.2
10  3  1  0.1
4   2  0  0.5
0   1  1  0.1
2   1  1  0.3
>>> s.sample(n=6, random_state=1)
   A  B    C
2  1  1  0.3
3  1  1  0.4
4  2  0  0.5
9  3  0  0.9
1  1  0  0.2
6  2  1  0.7
>>> s.sample(frac=0.5, replace=True, random_state=1)
   A  B    C
5  2  0  0.6
8  3  0  0.8
9  3  0  0.9
5  2  0  0.6
0  1  1  0.1
0  1  1  0.1


>>> s.sample(n=2, axis=1)
    B  A
0   1  1
1   0  1
2   1  1
3   1  1
4   0  2
5   0  2
6   1  2
7   1  3
8   0  3
9   0  3
10  1  3
>>> s.sample(n=2, axis=0)
   A  B     C
7  3  1  0.34
4  2  0  0.50

