## 基于jieba分词的Textrank和IT-IDF
***
- 参考资料
  - [python结巴分词、jieba加载停用词表](http://blog.csdn.net/u012052268/article/details/77825981)
  
  - [利用jieba分词,构建词云图](http://www.jianshu.com/p/e3308090a8e0)

### 标注词性
***
```python
import jieba.posseg as pseg
words = pseg.cut("你我他爱北京天安门更爱长城")
for word, flag in words:
    # 格式化模版并传入参数
    print('%s, %s' % (word, flag))
```

### 手动添加和删除新词
***
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
***
```python
'''添加自定义字典'''
jieba.load_userdict(r'PATH\userdict.txt')
'''添加停用词'''
jieba.analyse.set_stop_words(r'PATH\stopped_words_two.txt')
```

### Tokenize：返回词语在原文的起止位置
***
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
***
> [简书](http://www.jianshu.com/p/e3308090a8e0)
> 
> [博客](http://www.cnblogs.com/csj007523/p/7773027.html)
## 关键词提取-bosonnlp
***
- 参考文献

  - [关键词提取](http://docs.bosonnlp.com/keywords.html)
  
  - [bosonnlp.py](http://bosonnlp-py.readthedocs.io/#bosonnlp.BosonNLP.extract_keywords)

### 词性表
***

- 名词 (1个一类，7个二类，5个三类)
```
n 名词
nr 人名
nr1 汉语姓氏
nr2 汉语名字
nrj 日语人名
nrf 音译人名
ns 地名
nsf 音译地名
nt 机构团体名
nz 其它专名
nl 名词性惯用语
ng 名词性语素
```

- 时间词(1个一类，1个二类)
```
t 时间词
tg 时间词性语素
```

- 处所词(1个一类)
```
s 处所词
```

- 方位词(1个一类)
```
f 方位词
```

- 动词(1个一类，9个二类)
```
v 动词
vd 副动词
vn 名动词
vshi 动词“是”
vyou 动词“有”
vf 趋向动词
vx 形式动词
vi 不及物动词（内动词）
vl 动词性惯用语
vg 动词性语素
``` 

- 形容词(1个一类，4个二类)
```
a 形容词
ad 副形词
an 名形词
ag 形容词性语素
al 形容词性惯用语
```

- 区别词(1个一类，2个二类)
```
b 区别词
bl 区别词性惯用语
```

- 状态词(1个一类)
```
z 状态词
```

- 代词(1个一类，4个二类，6个三类)
```
r 代词
rr 人称代词
rz 指示代词
rzt 时间指示代词
rzs 处所指示代词
rzv 谓词性指示代词
ry 疑问代词
ryt 时间疑问代词
rys 处所疑问代词
ryv 谓词性疑问代词
rg 代词性语素
```

- 数词(1个一类，1个二类)
```
m 数词
mq 数量词
```

- 量词(1个一类，2个二类)
```
q 量词
qv 动量词
qt 时量词
```

- 副词(1个一类)
```
d 副词
```

- 介词(1个一类，2个二类)
```
p 介词
pba 介词“把”
pbei 介词“被”
```

- 连词(1个一类，1个二类)
```
c 连词
cc 并列连词
```

- 助词(1个一类，15个二类)

```
u 助词
uzhe 着
ule 了 喽
uguo 过
ude1 的 底
ude2 地
ude3 得
usuo 所
udeng 等 等等 云云
uyy 一样 一般 似的 般
udh 的话
uls 来讲 来说 而言 说来
uzhi 之
ulian 连 （“连小学生都会”）
```

- 叹词(1个一类)
```
e 叹词
```

- 语气词(1个一类)
```
y 语气词(delete yg)
```

- 拟声词(1个一类)
```
o 拟声词
```

- 前缀(1个一类)
```
h 前缀
```

- 后缀(1个一类)
```
k 后缀
```

- 字符串(1个一类，2个二类)
```
x 字符串
xx 非语素字
xu 网址URL
```

- 标点符号(1个一类，16个二类)
```
w 标点符号
wkz 左括号，全角：（ 〔 ［ ｛ 《 【 〖 〈 半角：( [ { <
wky 右括号，全角：） 〕 ］ ｝ 》 】 〗 〉 半角： ) ] { >
wyz 左引号，全角：“ ‘ 『
wyy 右引号，全角：” ’ 』
wj 句号，全角：。
ww 问号，全角：？ 半角：?
wt 叹号，全角：！ 半角：!
wd 逗号，全角：， 半角：,
wf 分号，全角：； 半角： ;
wn 顿号，全角：、
wm 冒号，全角：： 半角： :
ws 省略号，全角：…… …
wp 破折号，全角：—— －－ ——－ 半角：--- ----
wb 百分号千分号，全角：％ ‰ 半角：%
wh 单位符号，全角：￥ ＄ ￡ ° ℃ 半角：$
```

### sklearn计算TF-IDF
***
```
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
              "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
              "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
              "我 爱 北京 天安门"]  # 第四类文本的切词结果
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            print word[j], weight[i][j]
```
