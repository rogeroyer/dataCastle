from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
X = iris.data
y = iris.target

'''
递归特征消除法，返回特征选择后的数据
参数estimator为基模型
参数n_features_to_select为选择的特征个数
'''
# result = RFE(estimator=LogisticRegression(), n_features_to_select=3).fit_transform(X_train, )
# print(result)

# gbdt_model = clf.fit(X_train, y_train)  # Training model
# predicty_x = gbdt_model.predict_proba(test1217_x)[:, 1]  # predict: probablity of 1

# 弱分类器的数目
n_estimator = 10
# 切分为测试集和训练集，比例0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
# 调用GBDT分类模型。
grd = GradientBoostingClassifier(n_estimators=n_estimator)
# 调用one-hot编码。
grd_enc = OneHotEncoder()
# 调用LR分类模型。
grd_lm = LogisticRegression()

'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(X_train, y_train)


# fit one-hot编码器
grd_enc.fit(grd.apply(X_train)[:, :, 0])
''' 
使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
'''

print(type(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0])))
print(grd_enc.transform(grd.apply(X_test)[:, :, 0]))

grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# 用训练好的LR模型多X_test做预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]

# 根据预测结果输出
# fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
# print(y_pred_grd_lm)
