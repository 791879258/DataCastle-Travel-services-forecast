#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-25 16:41:30
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.linear_model import LogisticRegression

# import the LogisticRegression
from sklearn.linear_model import LogisticRegression

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
train_data = pd.read_csv(project_path + 'preprocess/traindata_output/traindata_output.csv')
test_data = pd.read_csv(project_path + 'preprocess/testdata_output/testdata_output.csv')
# # 删除actionType1~actionType9
# train_data.drop(train_data.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9]], axis=1, inplace=True)
# test_data.drop(test_data.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9]], axis=1, inplace=True)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
train_y = train_data.orderType
del train_data['orderType']
train_X = train_data.iloc[:, 1:]
test_data_lr = test_data.iloc[:, 1:]

classifier = LogisticRegression()  # 使用类，参数全是默认的
classifier.fit(train_X, train_y)  # 训练数据来学习，不需要返回值

result = pd.DataFrame(classifier.predict(test_data_lr), columns=['orderType'])  # 测试数据，分类返回标记
result = pd.concat([test_data['userid'], result], axis=1)
result.to_csv(project_path + 'result/lr/' + time.strftime("%m-%d-%H-%M", time.localtime()) + '-result' + '.csv',
              index=False)
print(result)
