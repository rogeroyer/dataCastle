## TF-IDF（term frequency–inverse document frequency）
TF-IDF是一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随著它在文件中出现的次数成正比增加，但同时会随著它在语料库中出现的频率成反比下降。TF-IDF加权的各种形式常被搜寻引擎应用，作为文件与用户查询之间相关程度的度量或评级。除了TF-IDF以外，因特网上的搜寻引擎还会使用基于连结分析的评级方法，以确定文件在搜寻结果中出现的顺序。

> 在文本挖掘中，要对文本库分词，而分词后需要对个每个分词计算它的权重，而这个权重可以使用TF-IDF计算。 

###  TF(term frequency)
TF就是分词出现的频率：该分词在该文档中出现的频率，算法是：（该分词在该文档出现的次数）/ (该文档分词的总数)，这个值越大表示这个词越重要，即权重就越大。

> 例如：一篇文档分词后，总共有500个分词，而分词”Hello”出现的次数是20次，则TF值是： tf =20/500=2/50=0.04 

### IDF（inversedocument frequency）
IDF逆向文件频率,一个文档库中，一个分词出现在的文档数越少越能和其它文档区别开来。算法是：log((总文档数/出现该分词的文档数)+0.01) ；（注加上0.01是为了防止log计算返回值为0）。

> 例如：一个文档库中总共有50篇文档，2篇文档中出现过“Hello”分词，则idf是： Idf = log(50/2 + 0.01) = log(25.01)=1.39811369 TF-IDF结合计算就是 tf*idf,比如上面的“Hello”分词例子中： TF-IDF = tf* idf = (20/500)* log(50/2 + 0.01)= 0.04*1.39811369=0.0559245476
***
### [sklearn-API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.transform)
Examples:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.shape)
print(X)
```

```
OUTPUT:
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
(4, 9)
  (0, 8)	0.38408524091481483
  (0, 3)	0.38408524091481483
  (0, 6)	0.38408524091481483
  (0, 2)	0.5802858236844359
  (0, 1)	0.46979138557992045
  (1, 8)	0.281088674033753
  (1, 3)	0.281088674033753
  (1, 6)	0.281088674033753
  (1, 1)	0.6876235979836938
  (1, 5)	0.5386476208856763
  (2, 8)	0.267103787642168
  (2, 3)	0.267103787642168
  (2, 6)	0.267103787642168
  (2, 0)	0.511848512707169
  (2, 7)	0.511848512707169
  (2, 4)	0.511848512707169
  (3, 8)	0.38408524091481483
  (3, 3)	0.38408524091481483
  (3, 6)	0.38408524091481483
  (3, 2)	0.5802858236844359
  (3, 1)	0.46979138557992045
```
***
### Task2.1
代码示例：将属性 "article" 转化为 tf-idf 特征矩阵，调用的sklearn接口，我只读取了训练集前500条数据进行尝试以高效地进行代码测试。
```python
train_data = pd.read_csv('../dataSet/train_set.csv', encoding='utf-8', nrows=500, usecols=[1])
train_data['article_list'] = train_data['article'].map(lambda index: index.split(' '))
train_data['length'] = train_data['article_list'].map(lambda index: len(index))
print(train_data['length'].max())

# calc length of set[all words]
temp = set([])
train_data['max_name'] = train_data['article_list'].map(lambda index: set(index))
for i in range(len(train_data)):
    temp = temp | train_data.loc[i, 'max_name']
print(len(temp))

vectorizer = TfidfVectorizer(encoding='utf-8', )
result = vectorizer.fit_transform(train_data['article'])
print(type(result))    # <class 'scipy.sparse.csr.csr_matrix'>
result = result.A      # convert csr_matrix to ndarray matrix
print(result.shape)
print(result)          # features matrix

```

样例输出：
```
8453
4484
<class 'scipy.sparse.csr.csr_matrix'>
(500, 4484)
[[0.         0.         0.         ... 0.         0.         0.        ]
 [0.00819422 0.         0.         ... 0.         0.         0.        ]
 [0.00591508 0.         0.         ... 0.         0.         0.        ]
 ...
 [0.01155529 0.         0.         ... 0.         0.         0.        ]
 [0.03134187 0.         0.         ... 0.         0.         0.0174317 ]
 [0.         0.         0.         ... 0.         0.         0.        ]]
```

结果分析：
1. article属性中包含相同的字，即有重复数字。
2. TfidfVectorizer.fit_transform 输出类型为 'scipy.sparse.csr.csr_matrix'
3. 可通过 '.A' 将 'scipy.sparse.csr.csr_matrix' 转化为 'ndarray' 类型
4. 最后地输出即为每个样本的 tf-idf 特征值
