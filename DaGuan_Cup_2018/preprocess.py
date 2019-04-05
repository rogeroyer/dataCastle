# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


train_data = pd.read_csv('../dataSet/train_set.csv', encoding='utf-8', nrows=500)
print(train_data.columns)      # ['id', 'article', 'word_seg', 'class']
print(train_data['article'])
print(train_data['word_seg'])
print(train_data.info())
print(train_data.describe())


"""calc length of article and word_seg"""
train_data['article'] = train_data['article'].map(lambda index: index.split(' '))
train_data['article'] = train_data['article'].map(lambda index: len(index))
train_data['word_seg'] = train_data['word_seg'].map(lambda index: index.split(' '))
train_data['word_seg'] = train_data['word_seg'].map(lambda index: len(index))
print(train_data)


