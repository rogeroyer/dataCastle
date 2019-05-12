<h2 align="center">自然语言处理第七期</h2>

## 任务代码参考链接：
- imdb
    - https://tensorflow.google.cn/tutorials/keras/basic_text_classification
- thuc_news
    - https://blog.csdn.net/u011439796/article/details/77692621
    - https://github.com/gaussic/text-classification-cnn-rnn

### 文件介绍
- data
  - data_loader.py:为THUC新闻分类数据准备
- dataSet
  - imdb:存放imdb的数据集
  - thuc_news:存放THUC的数据集
- model
  - rnn_model.py:RNN模型类
  - cnn_model.py:CNN模型类
- imdb.py:imdb分类源代码
- thuc_news_cnn.py:THUC分类CNN模型
- thuc_news_rnn.py:THUC分类RNN模型

## 评价指标

### **名词解释**

- ROC曲线：Receiver Operating Characteristic Curve

- AUC：Area Under Curve
***
### **ROC曲线**
![ROC曲线](http://img.blog.csdn.net/20180303220440674)
> 分类模型尝试将各个实例（instance）划归到某个特定的类，而分类模型的结果一般是实数值，如逻辑回归，其结果是从0到1的实数值。这里就涉及到如何确定阈值（thresholdvalue）,使得模型结果大于这个值，划为一类，小于这个值，划归为另一类。
> 
> 考虑一个二分问题，即将实例分成正类（positive）或负类（negative）。对一个二分问题来说，会出现四种情况。如果一个实例是正类并且也被预测成正类，即为真正类（Truepositive）,如果实例是负类被预测成正类，称之为假正类（Falsepositive）。相应地，如果实例是负类被预测成负类，称之为真负类（True positive）,正类被预测成负类则为假负类（falsenegative）。

***
### **相关计算公式**

![计算公式](http://img.blog.csdn.net/20180303220425667)


> 从表引入两个新名词。其一是真正类率(true positive rate ,TPR), 计算公式为TPR=TP / (TP +FN)，刻画的是分类器所识别出的正实例占所有正实例的比例。另外一个是假正类率(false positive rate,FPR),计算公式为FPR= FP / (FP + TN)，计算的是分类器错认为正类的负实例占所有负实例的比例。还有一个真负类率（True Negative Rate，TNR），也称为specificity,计算公式为TNR=TN / (FP + TN) = 1 − FPR。
> 
> 在一个二分类模型中，对于所得到的连续结果，假设已确定一个阀值，比如说 0.6，大于这个值的实例划归为正类，小于这个值则划到负类中。如果减小阀值，减到0.5，固然能识别出更多的正类，也就是提高了识别出的正例占所有正例的比类，即TPR,但同时也将更多的负实例当作了正实例，即提高了FPR。为了形象化这一变化，在此引入ROC。

***
### **AUC值的计算**


> AUC（Area UnderCurve）被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。
> 
> 在了解了ROC曲线的构造过程后，编写代码实现并不是一件困难的事情。相比自己编写代码，有时候阅读其他人的代码收获更多，当然过程也更痛苦些。在此推荐[scikit-learn](http://scikit-learn.org/stable/)中关于计算[AUC](https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L39)的代码。

### **召回率(recall)**
> 计算公式：TP / (TP + FN), 即为在实际为坏人的人中，预测正确（预测为坏人）的人占比。

### **准确率(precision)**
> 计算公式：TP / (TP + FP), 即为在预测为坏人的人中，预测正确（实际为坏人）的人占比。

### **F1值**
> F1值是精确率和召回率的调和均值，即 F1 = 2*precision*recall / (precision+recall)，相当于精确率和召回率的综合评价指标。

### **P-R曲线**
> P-R曲线就是精确率precision vs 召回率recall 曲线，以recall作为横坐标轴，precision作为纵坐标轴。

***
### **分类模型评估**

<table align="center">
	<tr>
		<td>指标</td>
		<td>描述</td>
		<td>Scikit-learn函数</td>
	</tr>
	<tr>
		<td>Precision</td>
		<td>精准度</td>
		<td>from sklearn.metrics import precision_score</td>
	</tr>
	<tr>
		<td>Recall</td>
		<td>召回度</td>
		<td>from sklearn.metrics import recall_score</td>
	</tr>
	<tr>
		<td>F1</td>
		<td>	F1值</td>
		<td>from sklearn.metrics import f1_score</td>
	</tr>
	<tr>
		<td>Confusion Matrix</td>
		<td>混淆矩阵</td>
		<td>from sklearn.metrics import confusion_matrix</td>
	</tr>
	<tr>
		<td>ROC</td>
		<td>ROC曲线</td>
		<td>from sklearn.metrics import roc</td>
	</tr>
	<tr>
		<td>AUC</td>
		<td>ROC曲线下的面积</td>
		<td>from sklearn.metrics import auc</td>
	</tr>
</table>

***
### **计算方式**
![这里写图片描述](https://img-blog.csdn.net/20180414191247979)

### **python实现**

- 调用sklearn
```python
from sklearn.metrics import roc_auc_score
roc_auc_score(lise_one, list_two)  # list既可以是list，也可以是pandas.Series #
```

-  函数实现（复杂度较高）
```python
def cal_auc(list_one, list_two):
    '''计算AUC值'''
    positive = []
    negative = []
    for index in range(len(list_one)):
        if list_one[index] == 1:
            positive.append(index)
        else:
            negative.append(index)
    SUM = 0
    for i in positive:
        for j in negative:
            if list_two[i] > list_two[j]:
                SUM += 1
            elif list_two[i] == list_two[j]:
                SUM += 0.5
            else:
                pass
    return SUM / (len(positive)*len(negative))
```

### **参考资料**

- [`ROC曲线(Receiver Operating Characteristic Curve)`](http://blog.sina.com.cn/s/blog_531bb7630100v4ny.html)

- [`Wikipedia`](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

- [`ROC和AUC介绍以及如何计算AUC`](http://alexkong.net/2013/06/introduction-to-auc-and-roc/)

***
