# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:11:19 2018

@author: yuwei
"""


import pandas as pd

data1 = pd.read_csv('ans1.csv')
data2 = pd.read_csv('ans.csv')

minmin1, maxmax1 = min(data1['PROB']),max(data1['PROB'])
data1['PROB'] = data1['PROB'].map(lambda x:(x-minmin1)/(maxmax1-minmin1))

minmin2, maxmax2 = min(data2['PROB']),max(data2['PROB'])
data2['PROB'] = data2['PROB'].map(lambda x:(x-minmin2)/(maxmax2-minmin2))


data = data1

data['PROB'] = 0.8*data1['PROB'] + 0.2*data2['PROB']
data['PROB'] = data['PROB'].map(lambda x:'%.4f' % x)

data.to_csv('ans.csv',index=False)
