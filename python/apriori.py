
def Apriori(trans, support=0.01, minlen=1):
    ts = pd.get_dummies(trans.unstack().dropna()).groupby(level=1).sum()
    print(ts)
    collen, rowlen = ts.shape

    pattern = []
    for cnum in range(minlen, rowlen+1):
        for cols in combinations(ts, cnum):
            patsup = ts[list(cols)].all(axis=1).sum()
            patsup = float(patsup)/collen
            pattern.append([",".join(cols), patsup])
    sdf = pd.DataFrame(pattern, columns=["Pattern", "Support"])
    results = sdf[sdf.Support >= support]

    return results
   
   
user_register = pd.read_csv(path + 'user_register_log.txt', encoding='utf-8', sep='\t', header=None, names=['user_id', 'register_day', 'register_type', 'device_type'], dtype={0: np.str, 1: np.str, 2: np.str, 3: np.str})
trans = user_register.loc[0: 10][['register_type', 'device_type']]
print(trans)
result = Apriori(trans)
print(result)

output:
   register_type device_type
0              4         270
1              0           5
2              0           0
3              1          49
4              0          72
5              1        4912
6              0          11
7              1        1454
8              0          67
9              1           6
10             0         136
    0  1  11  136  1454  270  4  49  4912  5  6  67  72
0   0  0   0    0     0    1  1   0     0  0  0   0   0
1   1  0   0    0     0    0  0   0     0  1  0   0   0
2   2  0   0    0     0    0  0   0     0  0  0   0   0
3   0  1   0    0     0    0  0   1     0  0  0   0   0
4   1  0   0    0     0    0  0   0     0  0  0   0   1
5   0  1   0    0     0    0  0   0     1  0  0   0   0
6   1  0   1    0     0    0  0   0     0  0  0   0   0
7   0  1   0    0     1    0  0   0     0  0  0   0   0
8   1  0   0    0     0    0  0   0     0  0  0   1   0
9   0  1   0    0     0    0  0   0     0  0  1   0   0
10  1  0   0    1     0    0  0   0     0  0  0   0   0
   Pattern   Support
0        0  0.545455
1        1  0.363636
2       11  0.090909
3      136  0.090909
4     1454  0.090909
5      270  0.090909
6        4  0.090909
7       49  0.090909
8     4912  0.090909
9        5  0.090909
10       6  0.090909
11      67  0.090909
12      72  0.090909
14    0,11  0.090909
15   0,136  0.090909
21     0,5  0.090909
23    0,67  0.090909
24    0,72  0.090909
27  1,1454  0.090909
30    1,49  0.090909
31  1,4912  0.090909
33     1,6  0.090909
63   270,4  0.090909

