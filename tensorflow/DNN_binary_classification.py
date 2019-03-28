# -*- coding:utf-8 -*-

import os
import time
import math
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(2019)
submit_path = './/submit//'


def calc_auc_score(validate, prediction):
    """
    :param validate:DataFrame columns=['Coupon_id', 'label]
    :param prediction:list
    :return:validate auc score
    """
    prediction = pd.DataFrame(prediction, columns=['prob'])
    validate = pd.concat([validate, prediction], axis=1)
    aucs = 0
    lens = 0
    for name, group in validate.groupby('Coupon_id'):
        if len(set(list(group['label']))) == 1:
            continue
        aucs += roc_auc_score(group['label'], group['prob'])
        lens += 1
    auc = aucs / lens
    return auc


def load_data():
    # Read data
    train_feature = pd.read_csv('../model_two/features/train_feature.csv', encoding='utf-8', low_memory=False).fillna(0)
    validate_feature = pd.read_csv('../model_two/features/validate_feature.csv', encoding='utf-8', low_memory=False).fillna(0)
    test_feature = pd.read_csv('../model_two/features/test_feature.csv', encoding='utf-8', low_memory=False).fillna(0)

    train_label = train_feature['label'].values.reshape(-1, 1)
    train_feature = train_feature.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Date', 'label'], axis=1).values
    validate_label = validate_feature['label'].values.reshape(-1, 1)
    validate_feature = validate_feature.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Date', 'label'], axis=1).values
    test_feature = test_feature.drop(['User_id', 'Merchant_id', 'Coupon_id'], axis=1).values

    """data normalization"""
    scalar = MinMaxScaler()
    train_feature = scalar.fit_transform(train_feature)
    validate_feature = scalar.fit_transform(validate_feature)
    test_feature = scalar.fit_transform(test_feature)
    return train_feature, train_label, validate_feature, validate_label, test_feature


def network(x_shape, y_shape):
    ops.reset_default_graph()   # reset computation graph
    """
    x_shape: input dim
    y_shape: output dim
    """
    x = tf.placeholder(tf.float32, shape=(None, x_shape), name="x")
    y = tf.placeholder(tf.float32, shape=(None, y_shape), name="y")

    # flatten the input
    print(x)
    m = tf.layers.dense(x, units=256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    print(m)
    m = tf.layers.dropout(inputs=m, rate=0.4)
    print(m)
    m = tf.layers.dense(m, units=128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    print(m)
    #     m = tf.layers.dropout(inputs=m, rate=0.4)
    #     print(m)
    m = tf.layers.dense(m, units=64, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    print(m)
    #     m = tf.layers.dropout(inputs=m, rate=0.4)
    #     print(m)
    m = tf.layers.dense(m, units=32, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    print(m)
    #     m = tf.layers.dropout(inputs=m, rate=0.4)
    #     print(m)
    m = tf.layers.dense(m, units=16, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    print(m)
    #     m = tf.layers.dropout(inputs=m, rate=0.4)
    #     print(m)
    prediction = tf.layers.dense(m, units=y_shape, name="p", kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer())
    print(prediction)

    return x, y, prediction


def optimization(logits, labels):
    """
    logits: pred value
    labels: real value
    """
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))     # 交叉熵之和
    optim = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    # optim = tf.contrib.opt.NadamOptimizer(learning_rate=1e-4).minimize(loss)
    return loss, optim


def random_mini_batches(x, y, mini_batch_size=64):
    m = x.shape[0]            # 特征维数
    mini_batches = []

    # shuffle x and y
    permutation = list(np.random.permutation(m))   # 置换函数：打乱行索引
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]

    # partition
    num_complete_mini_batches = int(math.ceil(m / mini_batch_size))    # ceil 返回x的最大值，即大于等于x的最小整数。
    for k in range(0, num_complete_mini_batches):
        mini_batch_x = shuffled_x[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_y = shuffled_y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)   # return a tuple
        mini_batches.append(mini_batch)
    return mini_batches


def assess_validate(validate_predict):
    validate = pd.read_csv('../model_two/features/validate_feature.csv', encoding='utf-8', low_memory=False)[['Coupon_id', 'label']]
    return calc_auc_score(validate, validate_predict)


def train(x_train, y_train, x_cv, y_cv, x_test, epochs=100, batch_size=1000, store_result=False):
    x_shape = x_train.shape
    y_shape = y_train.shape

    print("input: ", x_shape[1], " output: ", y_shape[1])
    x, y, pred = network(x_shape[1], y_shape[1])
    # loss, optim
    loss, optim = optimization(pred, y)
    # acc
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # accuracy = tf.metrics.mean_absolute_error(labels=y, predictions=pred)
    accuracy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
    # accuracy = tf.reduce_mean(tf.square(y - pred))
    # accuracy = tf.metrics.auc(labels=y, predictions=pred)

    # init tensorflow
    init = tf.global_variables_initializer()
    score = None
    best_score = 0
    best_submit = None
    with tf.Session() as sess:
        sess.run(init)       # initializes the variables created
        for epoch in range(epochs):
            epoch_cost = 0
            epoch_acc = 0
            num_mini_batches = math.ceil(x_shape[0] / batch_size)          # calculate iterations
            mini_batches = random_mini_batches(x_train, y_train)                # access batches
            for mini_batch in mini_batches:
                (mini_batch_x, mini_batch_y) = mini_batch                       # train network according to batches
                _, mini_batch_cost, p, mini_batch_acc = sess.run([optim, loss, pred, accuracy], feed_dict={x: mini_batch_x, y: mini_batch_y})
                # print("pred shape: ", p)
                epoch_cost += mini_batch_cost / num_mini_batches
                epoch_acc += mini_batch_acc / num_mini_batches

            print("cost after epoch %i :  %.3f" % (epoch + 1, epoch_cost), end=" ")
            # print("  train accuracy   :  %.3f" % epoch_acc, end="")
            # print("  cv accuracy   :  %.3f" % ( accuracy.eval({x: x_cv, y: y_cv}) / math.ceil(x_cv.shape[0] / batch_size) ))         # accuracy.eval   "like"   sess.run    Tensor.eval(feed_dict=None, session=None)
            validate_prob = sess.run(tf.nn.sigmoid(pred), feed_dict={x: x_cv, y: y_cv})
            score = assess_validate(validate_prob)
            print('validate auc score is {}'.format(score))
            if best_score > score:
                pass
            else:
                best_score = copy.deepcopy(score)
                best_submit = sess.run(tf.nn.sigmoid(pred), feed_dict={x: x_test, y: y_train})

        print("network trained")
        print('validate auc score is {}'.format(score))

        best_submit = pd.DataFrame(best_submit.reshape(1, -1)[0], columns=['best_submit'])
        print('best validate auc score is {}'.format(best_score))
        print("validate perform best prediction:\n", best_submit)

        prob = sess.run(tf.nn.sigmoid(pred), feed_dict={x: x_test, y: y_train})
        prob = pd.DataFrame(prob.reshape(1, -1)[0], columns=['prob'])
        print("finally prediction:\n", prob)

    if store_result is True:
        test_user = pd.read_csv('../dataSet/ccf_offline_stage1_test_revised.csv', encoding='utf-8')
        test_user = test_user[['User_id', 'Coupon_id', 'Date_received']]
        test_user['prob'] = [index for index in prob['prob']]
        test_user['best_submit'] = [index for index in best_submit['best_submit']]

        time_string = time.strftime('_%Y%m%d%H%M%S_', time.localtime(time.time()))
        file_name = 'submit_nn' + time_string + str(score)[:5] + '_' + str(best_score)[:5] + '.csv'
        test_user[['User_id', 'Coupon_id', 'Date_received', 'prob']].to_csv(submit_path + file_name, index=None, encoding='utf-8', header=None)
        test_user[['User_id', 'Coupon_id', 'Date_received', 'best_submit']].to_csv(submit_path + 'best_perform_' + file_name, index=None, encoding='utf-8', header=None)
        print('%s store successfully!' % file_name)


def main():
    x_train, y_train, x_cv, y_cv, x_test = load_data()
    train(x_train, y_train, x_cv, y_cv, x_test, epochs=10, batch_size=64, store_result=True)       # batch_size 不宜过大


if __name__ == '__main__':
    main()

"""
epochs=10   batch_size=64    validate:0.702281894101369
epochs=5   batch_size=64    validate:0.7083333673387422
"""

