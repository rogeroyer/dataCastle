### 基于jieba分词的Textrank和IT-IDF

- 参考资料
  - [python结巴分词、jieba加载停用词表](http://blog.csdn.net/u012052268/article/details/77825981)

### 标注词性
```python
import jieba.posseg as pseg
words = pseg.cut("你我他爱北京天安门更爱长城")
for word, flag in words:
    # 格式化模版并传入参数
    print('%s, %s' % (word, flag))
```
