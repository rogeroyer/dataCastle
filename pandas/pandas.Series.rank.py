example:
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
print(type(obj.rank()))
print(obj.rank())
print (obj.rank(method = 'first',ascending=False))
print (obj.rank(method = 'max',ascending=False))
print (obj.rank(method = 'min',ascending=False))
result:
<class 'pandas.core.series.Series'>
0    6.5
1    1.0
2    6.5
3    4.5
4    3.0
5    2.0
6    4.5
dtype: float64
0    1.0
1    7.0
2    2.0
3    3.0
4    5.0
5    6.0
6    4.0
dtype: float64
0    2.0
1    7.0
2    2.0
3    4.0
4    5.0
5    6.0
6    4.0
dtype: float64
0    1.0
1    7.0
2    1.0
3    3.0
4    5.0
5    6.0
6    3.0
dtype: float64
  
  
  
  
  
  
  
  
# 排序特征提取 #
s = pd.DataFrame([['2012', 'A', 4], ['2012', 'B', 8], ['2011', 'A', 21], ['2011', 'B', 31]], columns=['Year', 'Manager', 'Return'])
b = pd.DataFrame([['2012', 'A', 3], ['2012', 'B', 7], ['2011', 'A', 20], ['2011', 'B', 30]], columns=['Year', 'Manager', 'Return'])
s = s.append(b)
print(s)
s.reset_index(drop=True, inplace=True)
print(s)
s['Rank'] = s.groupby(['Manager'])['Return'].rank(ascending=True)
print(s.sort_values(by=['Manager']))

result:
     Year Manager  Return
0  2012       A       4
1  2012       B       8
2  2011       A      21
3  2011       B      31
0  2012       A       3
1  2012       B       7
2  2011       A      20
3  2011       B      30
   Year Manager  Return
0  2012       A       4
1  2012       B       8
2  2011       A      21
3  2011       B      31
4  2012       A       3
5  2012       B       7
6  2011       A      20
7  2011       B      30
   Year Manager  Return  Rank
0  2012       A       4   2.0
2  2011       A      21   4.0
4  2012       A       3   1.0
6  2011       A      20   3.0
1  2012       B       8   2.0
3  2011       B      31   4.0
5  2012       B       7   1.0
7  2011       B      30   3.0
