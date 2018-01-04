### 结果分析

- XGBOOST model
  - 
  ```
  module = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=4,
        gamma=0.1,
        subsample=0.7,
        objective='binary:logistic',
        nthread=5,
        scale_pos_weight=1,
        # seed=27
    )
  ```
```
正确预测出的垃圾新闻数: 8616
算法输出的垃圾新闻数: 9965
数据集中原本的垃圾新闻数: 9067
P: 0.864626191670848
R: 0.9502591816477335
F1: 0.905422446406053
```

- GBDT model
  - `GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, subsample=0.7)`
```
正确预测出的垃圾新闻数: 8675
算法输出的垃圾新闻数: 10075
数据集中原本的垃圾新闻数: 9067
P: 0.8610421836228288
R: 0.9567662953567884
F1: 0.9063838679343852
```

[XGBoost | GBDT | LightGBM 区别与联系](http://www.cnblogs.com/mata123/p/7440774.html)
