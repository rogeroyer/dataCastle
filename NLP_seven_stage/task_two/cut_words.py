# -*- coding:utf-8 -*- 

import jieba
import jieba.analyse

seg_list = jieba.cut("我来到重庆市南岸区黄桷垭", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到重庆市南岸区黄桷垭", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了北京搜狐大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于清华大学，后在美国哈佛大学深造")  # 搜索引擎模式
print(", ".join(seg_list))


seg_list = jieba.analyse.extract_tags("小明硕士毕业于清华大学，后在美国哈佛大学深造，清华大学是千万学子心之所向。", topK=20)
print(seg_list)

seg_list = jieba.analyse.extract_tags("小明硕士毕业于清华大学，后在美国哈佛大学深造，清华大学是千万学子心之所向。", topK=20,withWeight=True)
print(seg_list)



""" 输出
Full Mode: 我/ 来到/ 重庆/ 重庆市/ 市南/ 南岸/ 南岸区/ 岸区/ 黄桷/ 垭
Default Mode: 我/ 来到/ 重庆市/ 南岸区/ 黄桷/ 垭
他, 来到, 了, 北京, 搜狐, 大厦
小明, 硕士, 毕业, 于, 清华, 华大, 大学, 清华大学, ，, 后, 在, 美国, 哈佛, 大学, 美国哈佛大学, 深造
['清华大学', '心之所', '小明', '学子', '美国哈佛大学', '深造', '硕士', '千万', '毕业']    (分词按照tfidf值排序)
[('清华大学', 1.616118944324), ('心之所', 1.19547675029), ('小明', 1.11280889297), ('学子', 0.987532596124), ('美国哈佛大学', 0.9741794568619999), ('深造', 0.90884932966), ('硕士', 0.887023973058), ('千万', 0.685751773599), ('毕业', 0.6231649363389999)]  (分词按照tfidf值排序且带权重)
"""
