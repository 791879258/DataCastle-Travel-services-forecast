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

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_path = project_path + "data/train/"
testdata_path = project_path + "data/test/"
analysis_future1_but_few_actionnum_path = project_path + 'preprocess/analysis_future1_but_few_actionnum/'

orderFuture_train = pd.read_csv(traindata_path + 'orderFuture_train.csv')
action_train = pd.read_csv(traindata_path + 'action_train.csv')
futureType1_userid_list = orderFuture_train[orderFuture_train['orderType'] == 1]['userid'].values.tolist()
threhold = 5
num1 = 0
num0 = 0
userid1 = []
userid0 = []
for userid in futureType1_userid_list:
    if (action_train[action_train['userid'] == userid].shape[0] <= threhold):
        num1 += 1
        userid1.append(userid)

for userid in action_train['userid'].unique():
    if (action_train[action_train['userid'] == userid].shape[0] <= threhold and
            orderFuture_train[orderFuture_train['userid'] == userid]['orderType'].values[0] == 0):
        num0 += 1
        userid0.append(userid)
print('futureType为1，有： ' + str(len(futureType1_userid_list)))
print('action_num<= ' + str(threhold) + '且futureType为1，有： ' + str(num1) + '人')
print('action_num<= ' + str(threhold) + '且futureType为0，有： ' + str(num0) + '人')

userid1_file = open(analysis_future1_but_few_actionnum_path + 'valid_userid_Type_true0_pred1.txt', 'w')
userid1_file.write("valid有13302 userid，以下249 userid真实orderType=0，但是预测为了1，即pred_proba>0.5\n\nuserid\n")
for item in userid1:
    userid1_file.write("%s\n" % item)
print(userid1)
print(userid0)
