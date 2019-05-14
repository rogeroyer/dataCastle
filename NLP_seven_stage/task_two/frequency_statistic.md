### sklearn CountVector
sklearn.feature_extraction.text.CountVector是sklearn.feature_extraction.text提供的文本特征提取方法的一种。

### 参数设置
sklearn.feature_extraction.text.CountVectorizer(
input=’content’,         #输入，可以是文件名字，文件，文本内容
encoding=’utf-8’,       #默认编码方式
decode_error=’strict’, # 编码错误的处理方式，有三种{'strict','ignore','replace}
strip_accents=None, # 去除音调，三种{'ascill','unicode',None},ascii处理的速度快，但只适用于ASCll编码，unicode适用于所有的字符，但速度慢
lowercase=True, # 转化为小写
preprocessor=None,
tokenizer=None, #
stop_words=None,
token_pattern=’(?u)\b\w\w+\b’, ngram_range=(1, 1),
analyzer=’word’, #停止词，一些特别多，但没有意义的词，例如 a ,the an
max_df=1.0,#
min_df=1, #词最少出现的次数
max_features=None,  #最大特征
vocabulary=None,
binary=False,
dtype=<class ‘numpy.int64’>)

### Example
```python
>>>from sklearn.feature_extraction.text import CountVectorizer
# 词袋模型
>>>vectorizer = CountVectorizer()
  corpus = [
  'This is the first document.',
  'This is the second second document.',
  'And the third one.',
  'Is this the first document?']
# 进行词袋处理
>>>X = vectorizer.fit_transform(corpus)
>>>print(X.toarray())
>>>
[[0 1 1 1 0 0 1 0 1]
 [0 1 0 1 0 2 1 0 1]
 [1 0 0 0 1 0 1 1 0]
 [0 1 1 1 0 0 1 0 1]]
# 获取的特征单词
>>>print(vectorizer.get_feature_names())
>>>['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
# 从结果可以看出，提取的词按字母顺序排列
# 默认的提取单词的长度至少为2，
>>>analyzer = vectorizer.build_analyzer()
>>>print(analyzer("This is a text document to analyze."))
>>>['this', 'is', 'text', 'document', 'to', 'analyze']
# 可以看出，字符长度为1的‘a','.'被过滤掉了
# 对新的文本进行处理
>>>vectorizer_result = vectorizer.transform(['Something completely new document.']).toarray()
>>>print(vectorizer_result)
>>>[[0 1 0 0 0 0 0 0 0]]

# 获取索引
>>>print(vectorizer.vocabulary_.get('document'))
>>>1
```
