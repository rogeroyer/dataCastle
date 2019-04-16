'''两种归一化方法'''
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
result = min_max_scaler.fit_transform(train_order_info[['amt_order', 'unit_price']])
print(pd.DataFrame(result))

train_order_info[['amt_order', 'unit_price']] = train_order_info[['amt_order', 'unit_price']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(train_order_info[['amt_order', 'unit_price']])

feature.iloc[:, 1:] = feature.iloc[:, 1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


>>> import numpy as np
>>> from sklearn.preprocessing import MinMaxScaler
>>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> data
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
>>> scaler = MinMaxScaler(feature_range=(0.01, 0.99))
>>> scaler.fit_transform(data)
array([[0.01, 0.01, 0.01],
       [0.5 , 0.5 , 0.5 ],
       [0.99, 0.99, 0.99]])
# 由此可见：归一化是针对每一列进行归一化的而不是全局归一化
