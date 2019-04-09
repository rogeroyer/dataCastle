# -*- coding:utf-8 -*-

import pandas as pd
from gensim.models import Word2Vec

""" example
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(min_count=1, size=10)    # size:the number of hidden layer neurons
model.build_vocab(sentences)      # prepare the model vocabulary
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors
vector = model.wv['dog']          # numpy vector of a word
print(vector)
"""


train = pd.read_csv('..//dataSet//train_set.csv', encoding='utf-8', engine='c', usecols=[1], nrows=10)
test = pd.read_csv('..//dataSet//test_set.csv', encoding='utf-8', engine='c', usecols=[1], nrows=10)
train['article'] = train['article'].map(lambda index: index.strip().split())
test['article'] = test['article'].map(lambda index: index.strip().split())
sentences = list(train['article']) + list(test['article'])
print('prepare corpus over.')

model = Word2Vec(size=100, window=5, min_count=5, workers=5, sg=0, iter=5)
model.build_vocab(sentences)      # prepare the model vocabulary
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors

# model = Word2Vec(sentences=sentences, size=100, window=5, min_count=5, workers=8, sg=0, iter=5)      # from two choose one

print('model training over.')

print(model.wv['306028'])     # numpy vector of a word - "306028"
print(model.wv.index2word)    # show key words
print(len(model.wv.vectors))  # show vector of all keys
print(model.corpus_count)     # number of corpus
