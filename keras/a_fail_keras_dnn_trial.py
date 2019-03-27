# -*- coding:utf-8 -*-

import time
import keras
import numpy as np
import pandas as pd
from keras import models
from keras import layers
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

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

    train_label = train_feature['label'].values
    train_feature = train_feature.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Date', 'label'], axis=1).values
    validate_label = validate_feature['label'].values
    validate_feature = validate_feature.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Date', 'label'], axis=1).values
    test_feature = test_feature.drop(['User_id', 'Merchant_id', 'Coupon_id'], axis=1).values

    scalar = MinMaxScaler()
    train_feature = scalar.fit_transform(train_feature)
    validate_feature = scalar.fit_transform(validate_feature)
    test_feature = scalar.fit_transform(test_feature)

    return train_feature, train_label, validate_feature, validate_label, test_feature


def build_model(columns):
    model = models.Sequential()
    """
    # Input - layer
    model.add(layers.Dense(100, activation="relu", input_shape=(columns, ), W_regularizer=l2()))
    # Hidden - layer
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output - layer
    model.add(layers.Dense(1, activation="sigmoid"))
    """

    model.add(Dense(200, input_dim=columns, activation="tanh", W_regularizer=l2()))
    model.add(Dropout(0.4))
    # model.add(Activation('tanh'))  # tanh #
    model.add(Dense(40, activation="relu", W_regularizer=l2()))     # tanh    relu
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  # sigmoid #

    model.summary()

    """
    # Compiling the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    """

    model.compile(optimizer='sgd', loss=keras.losses.binary_crossentropy)     # "rmsprop  adam"    "binary_crossentropy mse"
    return model


def fit_and_predict_model(train_feature, train_label, validate_feature, validate_label, test_feature, store_result=False):
    model = build_model(train_feature.shape[1])
    # model.fit(train_feature, train_label, epochs=1, batch_size=500, validation_data=(validate_feature, validate_label))
    early_stopping = EarlyStopping(monitor='loss', patience=50)  # val_loss
    model.fit(train_feature, train_label, batch_size=1000, epochs=1000, verbose=2, validation_data=[validate_feature, validate_label], callbacks=[early_stopping])    # , validation_data=[te_x, te_y] #
    prediction = list(model.predict(test_feature).reshape(-1, 1))
    prediction = pd.DataFrame({'prob': prediction})
    prediction['prob'] = prediction['prob'].map(lambda index: index[0])

    # validate
    validate = pd.read_csv('../model_two/features/validate_feature.csv', encoding='utf-8', low_memory=False)[['Coupon_id', 'label']]
    validate_predict = model.predict(validate_feature)
    print("Validate result:", validate_predict)
    print('validate auc score is {}'.format(calc_auc_score(validate, validate_predict)))
    print("Test result", prediction)

    if store_result is True:
        test_user = pd.read_csv('../dataSet/ccf_offline_stage1_test_revised.csv', encoding='utf-8')
        test_user = test_user[['User_id', 'Coupon_id', 'Date_received']]
        test_user['Probability'] = [index for index in prediction['prob']]

        time_string = time.strftime('_%Y%m%d%H%M%S', time.localtime(time.time()))
        file_name = 'submit_nn' + time_string + '.csv'
        test_user.to_csv(submit_path + file_name, index=None, encoding='utf-8', header=None)
        print('%s store successfully!' % file_name)


def main():
    train_feature, train_label, validate_feature, validate_label, test_feature = load_data()
    fit_and_predict_model(train_feature, train_label, validate_feature, validate_label, test_feature, store_result=False)


if __name__ == '__main__':
    main()

