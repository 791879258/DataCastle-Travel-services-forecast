#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2098-01-17 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import gc
from sklearn.model_selection import train_test_split

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_path = project_path + "data/train/"
traindata_path = project_path + "data/train/"
testdata_path = project_path + "data/test/"
feature_file_path = project_path + 'preprocess/feature_file/'
traindata_output_path = project_path + "preprocess/traindata_output/"
testdata_output_path = project_path + "preprocess/testdata_output/"

traindata_output = pd.read_csv(traindata_output_path + 'traindata_output.csv')
# testdata_output = pd.read_csv(testdata_output_path + 'testdata_output.csv')
train_y = traindata_output['orderType']
del traindata_output['orderType']
train_X = traindata_output
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)

traindata = X_train[
    ['userid', 'actionType1', 'actionType2', 'actionType3', 'actionType4', 'actionType5', 'actionType6', 'actionType7',
     'actionType8', 'actionType9']]
testdata = X_test[
    ['userid', 'actionType1', 'actionType2', 'actionType3', 'actionType4', 'actionType5', 'actionType6', 'actionType7',
     'actionType8', 'actionType9']]
testuserid_trainuserid_simivalue_orderType = defaultdict(lambda: (defaultdict(lambda: 0)))
train_userid_orderType_df = pd.concat([X_train['userid'], y_train], axis=1)
test_userid_orderType_df = pd.concat([X_test['userid'], y_test], axis=1)
# testdata = testdata.head(1)
testdata = testdata[testdata['userid'] == 110106933065]
for testuserid in testdata['userid']:
    num = 0
    for trainuserid in traindata['userid']:
        if (num % 1000 == 0):
            print(num)
        test_actionType = testdata[testdata['userid'] == testuserid].iloc[:, 1:].values
        train_actionType = traindata[traindata['userid'] == trainuserid].iloc[:, 1:].values
        value = float(np.sum(np.square(test_actionType - train_actionType))) / len(train_actionType)
        testuserid_trainuserid_simivalue_orderType[testuserid][trainuserid] = value
        # testuserid_trainuserid_simivalue_orderType[testuserid][trainuserid].append(
        #     train_userid_orderType_df[train_userid_orderType_df['userid'] == trainuserid]['orderType'].values[0])
        num += 1
    gc.collect()
# print(testuserid_trainuserid_simivalue_orderType)

TopK = 10
for id in testuserid_trainuserid_simivalue_orderType.keys():
    trainid_value = testuserid_trainuserid_simivalue_orderType[id]
    similar_userid_value = sorted(trainid_value.items(), key=lambda x: x[1])[:TopK]
    similar_userid_list = [i[0] for i in similar_userid_value]
    orderType_list = []
    TopK_userid_orderType = defaultdict(lambda: 0)
    for userid in similar_userid_list:
        orderType = train_userid_orderType_df[train_userid_orderType_df['userid'] == userid]['orderType'].values[0]
        TopK_userid_orderType[userid] = orderType
        orderType_list.append(orderType)
    num = 0
    for type in orderType_list:
        if (type != 0):
            num += 1
    print('离测试集' + str(id) + '最近的' + str(TopK) + '训练集user中,' + '有' + str(num) + '个类别为1，' + str(TopK - num) + '类别为0')
    print('其真实的orderType: ' + str(
        test_userid_orderType_df[test_userid_orderType_df['userid'] == id]['orderType'].values[0]))
    for userid in TopK_userid_orderType.keys():
        print(str(userid) + ': ' + str(TopK_userid_orderType[userid]))
