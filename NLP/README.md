## 基于jieba分词的Textrank和IT-IDF

- 参考资料
  - [python结巴分词、jieba加载停用词表](http://blog.csdn.net/u012052268/article/details/77825981)
  
  - [利用jieba分词,构建词云图](http://www.jianshu.com/p/e3308090a8e0)

### 标注词性
```python
import jieba.posseg as pseg
words = pseg.cut("你我他爱北京天安门更爱长城")
for word, flag in words:
    # 格式化模版并传入参数
    print('%s, %s' % (word, flag))
```

### 手动添加和删除新词
```python
import jieba
jieba.add_word('蓝瘦')
jieba.add_word('香菇')
jieba.add_word('我的小伙伴们')
jieba.add_word('我好方')
jieba.add_word('倒数第一')
jieba.del_word('累觉不爱')
jieba.del_word('很傻很天真')
jieba.del_word('何弃疗')
jieba.del_word('友谊的小船')
jieba.del_word('说翻就翻')
jieba.del_word('今天')

test_sent = (
"今天去食堂吃饭没有肉了，蓝瘦香菇\n"
"前天去爬山的时候脚崴了，结果还得继续出去工作，累觉不爱\n"
"你不是真的关心我，咱们俩友谊的小船说翻就翻\n"
"你真的是很傻很天真，我的小伙伴们都觉得你好傻\n"
"一不小心得了个全班倒数第一，我好方"
)
words = jieba.cut(test_sent)
print('/'.join(words))

result:
今/天/去/食堂/吃饭/没有/肉/了/，/蓝瘦/香菇/
Prefix dict has been built succesfully.
/前天/去/爬山/的/时候/脚崴/了/，/结果/还/得/继续/出去/工作/，/累觉/不爱/
/你/不是/真的/关心/我/，/咱们/俩/友谊/的/小船/说/翻/就/翻/
/你/真的/是/很傻/很/天真/，/我的小伙伴们/都/觉得/你好/傻/
/一不小心/得/了/个/全班/倒数第一/，/我好方
```

### 添加自定义词典和停用词
```python
'''添加自定义字典'''
jieba.load_userdict(r'PATH\userdict.txt')
'''添加停用词'''
jieba.analyse.set_stop_words(r'PATH\stopped_words_two.txt')
```

### Tokenize：返回词语在原文的起止位置
```python
# -*- coding: utf-8 -*-
import jieba
jieba.load_userdict("userdict.txt")
result = jieba.tokenize(u'我爱四川大学中快食堂的饭菜')
for tk in result:
  print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
result = jieba.tokenize(u'我爱四川大学中快食堂的饭菜', mode='search')
for tk in result:
  print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
```
### 鸣谢
> 來源：简书  链接：http://www.jianshu.com/p/e3308090a8e0

## 关键词提取-bosonnlp
- 参考文献

  - [关键词提取](http://docs.bosonnlp.com/keywords.html)
  
  - [bosonnlp.py](http://bosonnlp-py.readthedocs.io/#bosonnlp.BosonNLP.extract_keywords)
