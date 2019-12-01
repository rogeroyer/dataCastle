# Top 3 pandas functions you don't know about(probably)
>>> x.idxmin()
0
>>> x.idxmax()
4
>>> df = pd.DataFrame()
>>> df['X'] = [0, 0, 0, 0, 1, 2, 3, 4, 5]
>>> df['X'].ne(0)
0    False
1    False
2    False
3    False
4     True
5     True
6     True
7     True
8     True
Name: X, dtype: bool
>>> df['X'].ne(0).idxmax()
4
>>> df['X'].ne(0).idxmin()
0
>>> df = pd.DataFrame({'Name': ['Bob', 'Mark', 'Steph', 'Jess', 'Becky'], 'Points': [55, 98, 46, 77, 81]})
>>> df
    Name  Points
0    Bob      55
1   Mark      98
2  Steph      46
3   Jess      77
4  Becky      81
>>> df.nsmallest(3, 'Points')
    Name  Points
2  Steph      46
0    Bob      55
3   Jess      77
>>> df.nlargest(3, 'Points')
    Name  Points
1   Mark      98
4  Becky      81
3   Jess      77
>>>
