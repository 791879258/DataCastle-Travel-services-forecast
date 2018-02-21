#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-29 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_path = project_path + "data/train/"
xgb_valid = pd.read_csv(project_path + 'result/xgb_valid/01-17-15-01-result.csv')
analysis_future1_but_few_actionnum_path = project_path + 'preprocess/analysis_future1_but_few_actionnum/'
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")

num0 = 0
num1 = 0
userid0 = []
userid1 = []
for userid in xgb_valid['userid'].values:
    orderType_true = xgb_valid[xgb_valid['userid'] == userid]['orderType_true'].values[0]
    orderType_pred = xgb_valid[xgb_valid['userid'] == userid]['orderType_pred'].values[0]
    if (orderType_true == 1 and orderType_pred < 0.5):
        num1 += 1
        userid1.append(userid)
    if (orderType_true == 0 and orderType_pred > 0.5):
        num0 += 1
        userid0.append(userid)

print(xgb_valid.shape[0])
print('true0_pred1: ' + str(num0))
print('true1_pred0: ' + str(num1))
userid_orderHistory_arr = orderHistory_train['userid'].values
num00 = 0
for userid in userid0:
    if (userid in userid_orderHistory_arr):
        num00 += 1
num11 = 0
for userid in userid1:
    if (userid in userid_orderHistory_arr):
        num11 += 1

print('在分错的' + str(num0) + '个true0_pred1中，存在order_history有： ' + str(num00))
print('在分错的' + str(num1) + '个true1_pred0中，存在order_history有： ' + str(num11))

###############################
# userid1_file = open(analysis_future1_but_few_actionnum_path + 'valid_userid_Type_true1_pred0.txt', 'w')
# for item in userid1:
#     userid1_file.write("%s\n" % item)
#
# userid0_file = open(analysis_future1_but_few_actionnum_path + 'valid_userid_Type_true0_pred1.txt', 'w')
# for item in userid0:
#     userid0_file.write("%s\n" % item)
