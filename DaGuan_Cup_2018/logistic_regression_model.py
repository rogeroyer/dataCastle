# -*- coding:utf-8 -*-

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore')
submit_file = '..//submit//'


class LogisticRegressionModel(object):
    def __init__(self, max_iter=200, store_submit=True, rows=1000, dimensions=100):
        self.max_iter = max_iter
        self.store_submit = store_submit
        if rows == -1:
            self.rows = None
        else:
            self.rows = rows
        self.dimensions = dimensions

    def __load_data(self):
        train = pd.read_csv('../dataSet/train_set.csv', encoding='utf-8', nrows=self.rows)
        test = pd.read_csv('../dataSet/test_set.csv', encoding='utf-8', nrows=self.rows)
        return train, test

    def extract_tf_idf_feature(self):
        train, test = self.__load_data()
        print('load data over.')
        print('start to extract tf-idf features.')
        vector_article = TfidfVectorizer()
        article = pd.concat([train[['article']], test[['article']]], axis=0)
        vector_article.fit(article['article'])
        article_train_tf_idf_features = vector_article.transform(train['article']).A
        article_test_tf_idf_features = vector_article.transform(test['article']).A

        # vector_word_seg = TfidfVectorizer()
        # word_seg = pd.concat([train[['word_seg']], test[['word_seg']]], axis=0)
        # vector_word_seg.fit(word_seg['word_seg'])
        # word_seg_train_tf_idf_features = vector_word_seg.transform(train['word_seg']).A
        # word_seg_test_tf_idf_features = vector_word_seg.transform(test['word_seg']).A
        # train_feature = np.concatenate((article_train_tf_idf_features, word_seg_train_tf_idf_features), axis=1)
        # test_feature = np.concatenate((article_test_tf_idf_features, word_seg_test_tf_idf_features), axis=1)

        train_feature = article_train_tf_idf_features
        test_feature = article_test_tf_idf_features

        train_label = train[['class']].values
        print('extract features over.')
        print('shape of features matrix is {}'.format(train_feature.shape))
        return train_feature, test_feature, train_label, test[['id']]

    def __dimensions_reduction(self, train, test):
        pca = PCA(n_components=self.dimensions)
        pca.fit(np.concatenate((train, test), axis=0))
        train = pca.transform(train)
        test = pca.transform(test)
        # print(pca.explained_variance_ratio_)
        # print(pca.explained_variance_)
        print('shape of features matrix after dimension reduction is {}'.format(train.shape))
        return train, test

    def multi_class_model(self):
        train_feature, test_feature, train_label, submit = self.extract_tf_idf_feature()
        print('start to reduce dimensions.')
        train_feature, test_feature = self.__dimensions_reduction(train_feature, test_feature)
        print('reducing dimensions over.')
        print('start to train model.')
        model = LogisticRegression(penalty='l2', random_state=2019, solver='sag', max_iter=self.max_iter, multi_class='multinomial', n_jobs=4)
        model.fit(train_feature, train_label)
        print('training model over.')
        test = model.predict(test_feature)
        submit['class'] = [index for index in test]
        print(submit)
        if self.store_submit is True:
            time_string = time.strftime('_%Y%m%d%H%M%S_', time.localtime(time.time()))
            file_name = 'submit_lr' + time_string + '.csv'
            submit.to_csv(submit_file + file_name, index=None, encoding='utf-8')


if __name__ == '__main__':
    lr_model = LogisticRegressionModel(max_iter=500, rows=-1, dimensions=1000, store_submit=True)
    lr_model.multi_class_model()


