#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-10 10:16:20
# @Author  : guanglinzhou (xdzgl812@163.com)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
import time

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
train_data = pd.read_csv(project_path + 'preprocess/traindata_output/traindata_output.csv')
test_data = pd.read_csv(project_path + 'preprocess/testdata_output/testdata_output.csv')
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

train_y = train_data.orderType
del train_data['orderType']
train_X = train_data.iloc[:, 1:]
test_data_rf = test_data.iloc[:, 1:]

rf1 = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=None,
                             min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf=0.0,
                             max_features='auto',
                             max_leaf_nodes=5, bootstrap=True,
                             oob_score=True, n_jobs=4, random_state=None, verbose=1, warm_start=False,
                             class_weight=None)
rf1.fit(train_X, train_y)
pred = rf1.predict_proba(test_data_rf)
pred1 = [i[1] for i in pred]
result = pd.DataFrame(columns=['userid', 'orderType'])
result['userid'] = test_data['userid']
result['orderType'] = pred1
result.to_csv(project_path + 'result/rf/' + time.strftime("%m-%d-%H-%M", time.localtime()) + '-result' + '.csv',
              index=False)
