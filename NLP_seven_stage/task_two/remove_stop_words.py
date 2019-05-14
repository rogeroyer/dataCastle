# -*- coding:utf-8 -*-

import jieba

stopwords = [line.strip() for line in open(data_path + 'stop_words.txt', 'r', encoding='utf-8').readlines()]
stopwords = list(set(stopwords))

def segment_sentence(sentence):
    sentence_seg = jieba.cut(sentence.strip(), cut_all=False)
    out_str = []
    for word in sentence_seg:
        if word not in stopwords:
            if (word != '\t') and (word != '\n'):
                out_str.append(word)
    return out_str

if __name__ == '__main__':
  string = '今天天气真好啊，好像出去打篮球。'
  words = segment_sentence(string)
  print(words)
