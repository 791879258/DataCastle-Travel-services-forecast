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

project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'
traindata_path = project_path + "data/train/"
testdata_path = project_path + "data/test/"
analysis_future1_but_few_actionnum_path = project_path + 'preprocess/analysis_future1_but_few_actionnum/'

orderFuture_train = pd.read_csv(traindata_path + 'orderFuture_train.csv')
action_train = pd.read_csv(traindata_path + 'action_train.csv')

# # 观察valid_userid_Type_true1_pred0中的userid有多少action_num<=5
# valid_userid_Type_true1_pred0 = pd.read_csv(
#     analysis_future1_but_few_actionnum_path + 'valid_userid_Type_true1_pred0.txt', sep=" ", header=None,
#     names=["userid"])
# valid_userid_Type_true1_pred0_action_num_less5_list = []
# for userid in valid_userid_Type_true1_pred0['userid']:
#     if (action_train[action_train['userid'] == userid].shape[0] <= 5):
#         valid_userid_Type_true1_pred0_action_num_less5_list.append(userid)
# ##########################################################################
# #  valid_userid_Type_true1_pred0的长度为633，而其中action_num<=5有103个。


#
#
#
#
# # 查看验证集中action_num<=5的userid当中，预测的混淆矩阵
# xgb_valid = pd.read_csv(project_path + 'result/xgb_valid/01-17-15-01-result.csv')
# valid_userid_action_num_less5 = []
# for userid in xgb_valid['userid']:
#     if (action_train[action_train['userid'] == userid].shape[0] <= 5):
#         valid_userid_action_num_less5.append(userid)
# num_true0_pred0 = 0
# num_true0_pred1 = 0
# num_true1_pred0 = 0
# num_true1_pred1 = 0
# userid_true0_pred1_list = []
# userid_true1_pred0_list = []
# for userid in valid_userid_action_num_less5:
#     type_true = xgb_valid[xgb_valid['userid'] == userid]['orderType_true'].values[0]
#     type_pre = xgb_valid[xgb_valid['userid'] == userid]['orderType_pred'].values[0]
#     if (type_true == 0 and type_pre < 0.5):
#         num_true0_pred0 += 1
#     elif (type_true == 0 and type_pre > 0.5):
#         num_true0_pred1 += 1
#         userid_true0_pred1_list.append(userid)
#     elif (type_true == 1 and type_pre < 0.5):
#         num_true1_pred0 += 1
#         userid_true1_pred0_list.append(userid)
#     elif (type_true == 1 and type_pre > 0.5):
#         num_true1_pred1 += 1
# print('在验证集action_num<=5的用户中：')
# print('num_true0_pred0: ' + str(num_true0_pred0))  # 1803
# print('num_true0_pred1: ' + str(num_true0_pred1))  # 36
# print('num_true1_pred0: ' + str(num_true1_pred0))  # 103
# print('num_true1_pred1: ' + str(num_true1_pred1))  # 86
# print('userid_true0_pred1_list:')
# print(userid_true0_pred1_list)
# print('userid_true1_pred0_list:')
# print(userid_true1_pred0_list)
