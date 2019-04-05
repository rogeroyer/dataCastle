# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


def divid_data_set(divid_rate):
    """
    divid_rate:train data set ratio.
    """
    data = pd.read_csv('../dataSet/train_set.csv', encoding='utf-8', nrows=500)
    data_length = len(data)
    train_data = data.iloc[:divid_rate * data_length, :]
    validate_data = data.iloc[divid_rate * data_length:, :]
    return train_data, validate_data
    

if __name__ == '__main__':
     divid_data_set(divid_rate=0.7)
    
