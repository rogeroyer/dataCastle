"""
Function:Compute pairwise correlation of columns, excluding NA/null values
"""

print(df)
"""
    report_date      purchase        redeem
0      20140901  1.900814e+08  1.877045e+08
1      20140902  3.085200e+08  3.180026e+08
2      20140903  2.902890e+08  2.829147e+08
3      20140904  1.831572e+08  2.021045e+08
4      20140905  1.930872e+08  2.247545e+08
5      20140906  2.197495e+08  2.408994e+08
6      20140907  2.228348e+08  2.875038e+08
7      20140908  2.342706e+08  2.593379e+08
8      20140909  2.619176e+08  3.027310e+08
9      20140910  2.224958e+08  2.445697e+08
10     20140911  2.086921e+08  2.257327e+08
11     20140912  2.040120e+08  2.320804e+08
12     20140913  1.989356e+08  2.212479e+08
13     20140914  1.850599e+08  2.436286e+08
14     20140915  2.203710e+08  2.758664e+08
15     20140916  2.337517e+08  3.097221e+08
16     20140917  2.770907e+08  2.888892e+08
17     20140918  1.981147e+08  2.256813e+08
18     20140919  1.949051e+08  2.380341e+08
19     20140920  1.796488e+08  2.287753e+08
20     20140921  2.333414e+08  2.489703e+08
21     20140922  2.237573e+08  2.280016e+08
22     20140923  2.621109e+08  2.950991e+08
23     20140924  2.512324e+08  2.866228e+08
24     20140925  1.786131e+08  2.064958e+08
25     20140926  2.009365e+08  1.819922e+08
26     20140927  2.099326e+08  2.166192e+08
27     20140928  2.660031e+08  2.295278e+08
28     20140929  1.949040e+08  2.185512e+08
29     20140930  2.574096e+08  3.016542e+08
"""

print(df.corr(method='pearson'))
"""
report_date  purchase    redeem
report_date     1.000000 -0.070013 -0.107875
purchase       -0.070013  1.000000  0.780072
redeem         -0.107875  0.780072  1.000000
"""

