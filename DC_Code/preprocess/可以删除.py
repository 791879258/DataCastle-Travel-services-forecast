#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-29 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# for i in range(1, 10):
#     print('\'to_closest_' + str(i) + '_time\'+\',\'+\'to_closest_' + str(i) + '_dist\'+\',\'+\'to_closest_' + str(
#         i) + '_time_interval_mean\'+\',\'' + '+\'to_closest_' + str(
#         i) + '_time_interval_var\'+\',\'+\'to_closest_' + str(
#         i) + '_time_interval_min\'+\',\'+\'to_closest_' + str(i) + '_time_interval_max\'+\',\'+\'to_closest_' + str(
#         i) + '_time_interval_median\'+\',\'+')

# for i in range(1, 10):
#     print('del traindata_output[\'actionType' + str(i) + '\']')
#     print('del testdata_output[\'actionType' + str(i) + '\']')
project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
xgb_valid = pd.read_csv(project_path + 'result/xgb_valid/01-17-15-01-result.csv')
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
print(userid0)
print(userid1)
print('\n-----------\n')
print(xgb_valid.shape[0])
print(num0)
print(num1)

# ##########
# action_train = pd.read_csv(project_path + 'data/train/action_train.csv')
# userid_typenum = defaultdict(lambda: 0)
# for userid in userid1:
#     num = action_train[action_train['userid'] == userid].shape[0]
#     userid_typenum[userid] = num
