# 加权平均结合sigmoid反函数

> 首先将各个模型的结果代入到sigmoid反函数中，然后得到其均值，对其结果使用sigmoid函数。相较于普通的加权平均，这种方法更适合于结果具有较小差异性的。

```python
def f(x):
    res=1/(1+np.e**(-x))
    return res

def f_ver(x):
    res=np.log(x/(1-x))
    return res
```

> 经过队内讨论后，使用stacking结合sigmoid反函数的方法进行融合。用多个模型的结果作为新的特征进行训练，然后利用不同折数加参数，特征，样本（随机数种子）扰动，再使用加权平均结合sigmoid反函数得到最终成绩。

```python
skf=list(StratifiedKFold(y_loc_train, n_folds=10, shuffle=True, random_state=1024))
for i, (train, test) in enumerate(skf):
    print("Fold", i)
    model.fit(X_loc_train[train], y_loc_train[train], eval_metric='logloss',/
              eval_set=[(X_loc_train[train], y_loc_train[train]), (X_loc_train[test], /
              y_loc_train[test])],early_stopping_rounds=100)    
    test_pred= model.predict_proba(X_loc_test, num_iteration=-1)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
for i in range(10):
    res['predicted_score'] += res['prob_%s' % str(i)].apply(lambda x: math.log(x/(1-x)))
res['predicted_score'] = (res['predicted_score']/10).apply(lambda x: 1/(1+math.exp(-x)))
```
