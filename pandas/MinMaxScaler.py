'''两种归一化方法'''
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
result = min_max_scaler.fit_transform(train_order_info[['amt_order', 'unit_price']])
print(pd.DataFrame(result))

train_order_info[['amt_order', 'unit_price']] = train_order_info[['amt_order', 'unit_price']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(train_order_info[['amt_order', 'unit_price']])

feature.iloc[:, 1:] = feature.iloc[:, 1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
