# <p align="center">[IJCAL-2018 ](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100071.5678.2.f1b27cf0n8lDs8&raceId=231647)</p>

***

### 格式转换
```python
import pandas as pd

'''去掉辅助字母"a"  '''
test_original_data = pd.read_csv('D:/dataSet/IJCAI2018/round1_ijcai_18_test_a_20180301.csv', low_memory=False)
train_original_data = pd.read_csv('D:/dataSet/IJCAI2018/round1_ijcai_18_train_20180301.csv', low_memory=False)

test_columns = ['instance_id', 'item_id', 'item_category_list', 'item_property_list',
       'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
       'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id',
       'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id',
       'context_timestamp', 'context_page_id', 'predict_category_property',
       'shop_id', 'shop_review_num_level', 'shop_review_positive_rate',
       'shop_star_level', 'shop_score_service', 'shop_score_delivery']

train_columns = ['instance_id', 'item_id', 'item_category_list', 'item_property_list',
       'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
       'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id',
       'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id',
       'context_timestamp', 'context_page_id', 'predict_category_property',
       'shop_id', 'shop_review_num_level', 'shop_review_positive_rate',
       'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

def delete_a(data, columns):
    for index in columns:
        data[index] = [string.replace('a', '') for string in data[index]]
    return data

train_original_data = delete_a(train_original_data, train_columns)
test_original_data = delete_a(test_original_data, test_columns)
print(train_original_data)
print(test_original_data)

# train_original_data.to_csv('ijcai_train_data.csv', index=None)
# test_original_data.to_csv('ijcai_test_data.csv', index=None)
# 到此数据即可使用 #


##### 处理原数据 将分隔符' ' -> ',' ####
'''测试集'''
# f = open("D:/dataSet/IJCAI2018/round1_ijcai_18_test_a_20180301_oringinal.txt", 'r')             # 返回一个文件对象
# r = open("D:/dataSet/IJCAI2018/round1_ijcai_18_test_a_20180301.txt", 'w')
# line = f.readline()             # 调用文件的 readline()方法
# while line:
#     line = line.replace(',', '/')
#     line = line.replace(' ', 'a,')
#     # line = line.replace(':', ',')
#     r.write(line)
#     # print(line, end='')    # 在 Python 3中使用#
#     line = f.readline()
# f.close()
# r.close()
'''训练集'''
# f = open("D:/dataSet/IJCAI2018/round1_ijcai_18_train_20180301_original.txt", 'r')             # 返回一个文件对象
# r = open("D:/dataSet/IJCAI2018/round1_ijcai_18_train_20180301.txt", 'w')
# line = f.readline()             # 调用文件的 readline()方法
# while line:
#     line = line.replace(',', '/')
#     line = line.replace(' ', 'a,')
#     # line = line.replace(':', ',')
#     r.write(line)
#     # print(line, end='')    # 在 Python 3中使用#
#     line = f.readline()
# f.close()
# r.close()
```

### [评价指标](http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#sphx-glr-auto-examples-calibration-plot-calibration-multiclass-py)

```python
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

np.random.seed(0)

# Generate data
X, y = make_blobs(n_samples=1000, n_features=2, random_state=42,
                  cluster_std=5.0)
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)
```
       - log_loss python实现
 ```python
     def logloss(act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1 - epsilon, pred)
        ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
        ll = ll * -1.0 / len(act)
        return ll
    score = logloss(validate_label.values, validate_label_predict.values)   #   0.0816689217692  #
 ```
