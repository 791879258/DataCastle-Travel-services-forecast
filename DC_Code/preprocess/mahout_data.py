#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-28 22:38:41
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : ${link}
# @Version : $Id$


import os
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_path = project_path + "data/train/"
traindata_output_path = project_path + "preprocess/"
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderFuture_train = pd.read_csv(traindata_path + "orderFuture_train.csv")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
userComment_train = pd.read_csv(traindata_path + "userComment_train.csv")
userProfile_train = pd.read_csv(traindata_path + "userProfile_train.csv")
testdata_path = project_path + "data/test/"
testdata_output_path = project_path + "preprocess/"
action_test = pd.read_csv(testdata_path + "action_test.csv")
orderFuture_test = pd.read_csv(testdata_path + "orderFuture_test.csv")
orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")
userComment_test = pd.read_csv(testdata_path + "userComment_test.csv")
userProfile_test = pd.read_csv(testdata_path + "userProfile_test.csv")

# train
# userid itemid value
# itemid:0-9：行为，10：普通服务，11：精品服务
# value 行为次数和订单次数

# # action_train = action_train[action_train['userid'] == 100000000013]
# file = open(project_path + 'preprocess/middle_file/mahout_all_data.csv', 'w')
# file.write('userid,itemid,value')
#
#
# def func(df):
#     global count
#     global num
#     global userid_history
#     userid = df['userid'].values[0]
#     itemid = 0
#     value = 0
#     num += 1
#     if (num % 1000 == 0):
#         print(num)
#     if (count != 0):
#         df = df.groupby('actionType').count()
#         del df['actionTime']
#         df.rename(columns={'userid': 'count'}, inplace=True)
#         df.reset_index(inplace=True)
#         for index, row in df.iterrows():
#             trainfile.write('\n')
#             trainfile.write(str(userid) + ',' + str(row['actionType']) + ',' + str(row['count']))
#             file.write('\n')
#             file.write(str(userid) + ',' + str(row['actionType']) + ',' + str(row['count']))
#         # df['value'] = df['actionType'] * df['count']
#         # value = df['value'].sum()
#         if (userid in userid_history):
#             orderType_arr = orderHistory_train[orderHistory_train['userid'] == userid]['orderType'].values
#             value0 = 0
#             value1 = 0
#
#             for type in orderType_arr:
#                 if (type == 0):
#                     value0 += 1
#                 else:
#                     value1 += 1
#             trainfile.write('\n')
#             trainfile.write(str(userid) + ',' + str(10) + ',' + str(value0))
#             file.write('\n')
#             file.write(str(userid) + ',' + str(10) + ',' + str(value0))
#             if (value1 != 0):
#                 trainfile.write('\n')
#                 trainfile.write(str(userid) + ',' + str(11) + ',' + str(value1))
#                 file.write('\n')
#                 file.write(str(userid) + ',' + str(11) + ',' + str(value1))
#     count += 1
#     return df
#
#
# count = 0
# num = 0
# userid_history = orderHistory_train['userid'].values
# trainfile = open(project_path + 'preprocess/middle_file/mahout_data_train.csv', 'w')
# trainfile.write('userid,itemid,value')
# action_train.groupby('userid').apply(func)
#
#
# def func(df):
#     global count
#     global num
#     global userid_history
#     userid = df['userid'].values[0]
#     itemid = 0
#     value = 0
#     num += 1
#     if (num % 1000 == 0):
#         print(num)
#     if (count != 0):
#         df = df.groupby('actionType').count()
#         del df['actionTime']
#         df.rename(columns={'userid': 'count'}, inplace=True)
#         df.reset_index(inplace=True)
#         for index, row in df.iterrows():
#             testfile.write('\n')
#             testfile.write(str(userid) + ',' + str(row['actionType']) + ',' + str(row['count']))
#             file.write('\n')
#             file.write(str(userid) + ',' + str(row['actionType']) + ',' + str(row['count']))
#         # df['value'] = df['actionType'] * df['count']
#         # value = df['value'].sum()
#         if (userid in userid_history):
#             orderType_arr = orderHistory_test[orderHistory_test['userid'] == userid]['orderType'].values
#             value0 = 0
#             value1 = 0
#             for type in orderType_arr:
#                 if (type == 0):
#                     value0 += 1
#                 else:
#                     value1 += 1
#             testfile.write('\n')
#             testfile.write(str(userid) + ',' + str(10) + ',' + str(value0))
#             file.write('\n')
#             file.write(str(userid) + ',' + str(10) + ',' + str(value0))
#             if (value1 != 0):
#                 testfile.write('\n')
#                 testfile.write(str(userid) + ',' + str(11) + ',' + str(value1))
#                 file.write('\n')
#                 file.write(str(userid) + ',' + str(11) + ',' + str(value1))
#     count += 1
#     return df
#
#
# count = 0
# num = 0
# userid_history = orderHistory_test['userid'].values
# testfile = open(project_path + 'preprocess/middle_file/mahout_data_test.csv', 'w')
# testfile.write('userid,itemid,value')
# action_test.groupby('userid').apply(func)

user_list = []
user_list.extend(action_train['userid'].unique().tolist())
user_list.extend(action_test['userid'].unique().tolist())
userlist_file = open(project_path + 'preprocess/middle_file/mahout_userid.txt', 'w')
for item in user_list:
    userlist_file.write("%s\n" % item)
