#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-01 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import datetime
from dateutil.relativedelta import relativedelta
import gc
import time

################
project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_path = project_path + "data/train/"
traindata_output_path = project_path + "preprocess/"
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderFuture_train = pd.read_csv(traindata_path + "orderFuture_train.csv")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
testdata_path = project_path + "data/test/"
testdata_output_path = project_path + "preprocess/"
action_test = pd.read_csv(testdata_path + "action_test.csv")
orderFuture_test = pd.read_csv(testdata_path + "orderFuture_test.csv")
orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")

# 按照每个用户max(行为时间，历史订单时间)的日期为截止日期，往后一个月内作为题目所要求的未来一段时间。
# 现在构造特征，该一个月时间段内，总共的行为次数，订单次数。
# # train
# action_train['actionTime'] = action_train['actionTime'].apply(
#     lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
#
#
# # orderHistory_train用户的下单顺序有的不是按照时间先后排列的，需要先进行排序。
# def fun_history(df):
#     global cnt
#     if (cnt != 0):
#         time_list = df['orderTime'].values
#         time_list = sorted(time_list, key=int)
#         df['orderTime'] = time_list
#         return df
#     cnt += 1
#
#
# cnt = 0
# orderHistory_train = orderHistory_train.groupby('userid').apply(fun_history)
# orderHistory_train['orderTime'] = orderHistory_train['orderTime'].apply(
#     lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
#
# future_time_file = open(project_path + 'preprocess/feature_file/future_time_train.csv', 'w')
# future_time_file.write(
#     'userid,type1_num_month,type2_num_month,type3_num_month,type4_num_month,type5_num_month,type6_num_month,type7_num_month,type8_num_month,type9_num_month,type_total_num_month,orderType0_num_month,orderType1_num_month,orderType_total_num_month')
# # a = [100000000013]
# count = 0
# t0 = time.time()
# for userid in orderFuture_train['userid'].values:
#     # for userid in a:
#     count += 1
#     print(str(count) + ':   ' + str(userid))
#     future_time_file.write('\n')
#     future_time_file.write(str(userid))
#     max_datetime = max(action_train[action_train['userid'] == userid]['actionTime'].values[-1],
#                        orderHistory_train[orderHistory_train['userid'] == userid]['orderTime'].values[-1]) if (
#             len(orderHistory_train[orderHistory_train['userid'] == userid]['orderTime'].values) != 0) else \
#         action_train[action_train['userid'] == userid]['actionTime'].values[-1]
#     time_period = []
#     time_period.append(max_datetime)
#     u = datetime.datetime.strptime(max_datetime, '%Y-%m-%d')
#     d = relativedelta(months=1)
#     t = u + d
#     t = t.strftime('%Y-%m-%d')
#     time_period.append(t)
#     # 对应下面的1-9，arr[0]始终为0不影响后面求和
#     typeX_num_arr = np.zeros((10,))
#     typeX_num = 0
#     type_total_num = 0
#     orderTypeX_num_arr = np.zeros((2,))
#     orderTypeX_num = 0
#     orderType_total_num = 0
#
#     for userid in orderFuture_train['userid'].values:
#         action_train_user = action_train[action_train['userid'] == userid]
#         actionTime_max = action_train_user['actionTime'].values[-1]
#         actionTime_min = action_train_user['actionTime'].values[0]
#         # 无交集
#         if (actionTime_min > time_period[-1] or actionTime_max < time_period[0]):
#             pass
#         else:
#             action_train_user = action_train_user[
#                 (action_train_user['actionTime'] > time_period[0]) & (action_train_user['actionTime'] < time_period[1])]
#             df = action_train_user.groupby('actionType').count()
#             df.reset_index(inplace=True)
#             for i in range(1, 10):
#                 if (len(df[df['actionType'] == i]['actionTime'].values) != 0):
#                     typeX_num = df[df['actionType'] == i]['actionTime'].values[0]
#                     typeX_num_arr[i] += typeX_num
#         if (userid in orderHistory_train['userid'].values):
#             orderHistory_train_user = orderHistory_train[orderHistory_train['userid'] == userid]
#             orderTime_max = orderHistory_train_user['orderTime'].values[-1]
#             orderTime_min = orderHistory_train_user['orderTime'].values[0]
#             if (orderTime_min > time_period[-1] or orderTime_max < time_period[0]):
#                 pass
#             else:
#                 orderHistory_train_user = orderHistory_train_user[
#                     (orderHistory_train_user['orderTime'] > time_period[0]) & (orderHistory_train_user['orderTime'] <
#                                                                                time_period[1])]
#                 df = orderHistory_train_user.groupby('orderType').count()
#                 df.reset_index(inplace=True)
#                 for i in range(0, 2):
#                     if (len(df[df['orderType'] == i]['orderTime'].values) != 0):
#                         orderTypeX_num = df[df['orderType'] == i]['orderTime'].values[0]
#                         orderTypeX_num_arr[i] += orderTypeX_num
#     gc.collect()
#
#     type_total_num = np.sum(typeX_num_arr)
#     orderType_total_num = np.sum(orderTypeX_num_arr)
#     for i in range(1, 10):
#         future_time_file.write(',' + str(typeX_num_arr[i]))
#     future_time_file.write(',' + str(type_total_num))
#     for i in range(2):
#         future_time_file.write(',' + str(orderTypeX_num_arr[i]))
#     future_time_file.write(',' + str(orderType_total_num))
# print('train耗时: ' + str(time.time() - t0))

# test
action_test['actionTime'] = action_test['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))


# orderHistory_test用户的下单顺序有的不是按照时间先后排列的，需要先进行排序。
def fun_history(df):
    global cnt
    if (cnt != 0):
        time_list = df['orderTime'].values
        time_list = sorted(time_list, key=int)
        df['orderTime'] = time_list
        return df
    cnt += 1


cnt = 0
orderHistory_test = orderHistory_test.groupby('userid').apply(fun_history)
orderHistory_test['orderTime'] = orderHistory_test['orderTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))

future_time_file = open(project_path + 'preprocess/feature_file/future_time_test.csv', 'w')
future_time_file.write(
    'userid,type1_num_month,type2_num_month,type3_num_month,type4_num_month,type5_num_month,type6_num_month,type7_num_month,type8_num_month,type9_num_month,type_total_num_month,orderType0_num_month,orderType1_num_month,orderType_total_num_month')
# a = [100000000013]
count = 0
t0 = time.time()
for userid in orderFuture_test['userid'].values:
    # for userid in a:
    count += 1
    print(str(count) + ':   ' + str(userid))
    future_time_file.write('\n')
    future_time_file.write(str(userid))
    max_datetime = max(action_test[action_test['userid'] == userid]['actionTime'].values[-1],
                       orderHistory_test[orderHistory_test['userid'] == userid]['orderTime'].values[-1]) if (
            len(orderHistory_test[orderHistory_test['userid'] == userid]['orderTime'].values) != 0) else \
        action_test[action_test['userid'] == userid]['actionTime'].values[-1]
    time_period = []
    time_period.append(max_datetime)
    u = datetime.datetime.strptime(max_datetime, '%Y-%m-%d')
    d = relativedelta(months=1)
    t = u + d
    t = t.strftime('%Y-%m-%d')
    time_period.append(t)
    # 对应下面的1-9，arr[0]始终为0不影响后面求和
    typeX_num_arr = np.zeros((10,))
    typeX_num = 0
    type_total_num = 0
    orderTypeX_num_arr = np.zeros((2,))
    orderTypeX_num = 0
    orderType_total_num = 0

    for userid in orderFuture_test['userid'].values:
        action_test_user = action_test[action_test['userid'] == userid]
        actionTime_max = action_test_user['actionTime'].values[-1]
        actionTime_min = action_test_user['actionTime'].values[0]
        # 无交集
        if (actionTime_min > time_period[-1] or actionTime_max < time_period[0]):
            pass
        else:
            action_test_user = action_test_user[
                (action_test_user['actionTime'] > time_period[0]) & (action_test_user['actionTime'] < time_period[1])]
            df = action_test_user.groupby('actionType').count()
            df.reset_index(inplace=True)
            for i in range(1, 10):
                if (len(df[df['actionType'] == i]['actionTime'].values) != 0):
                    typeX_num = df[df['actionType'] == i]['actionTime'].values[0]
                    typeX_num_arr[i] += typeX_num
        if (userid in orderHistory_test['userid'].values):
            orderHistory_test_user = orderHistory_test[orderHistory_test['userid'] == userid]
            orderTime_max = orderHistory_test_user['orderTime'].values[-1]
            orderTime_min = orderHistory_test_user['orderTime'].values[0]
            if (orderTime_min > time_period[-1] or orderTime_max < time_period[0]):
                pass
            else:
                orderHistory_test_user = orderHistory_test_user[
                    (orderHistory_test_user['orderTime'] > time_period[0]) & (orderHistory_test_user['orderTime'] <
                                                                              time_period[1])]
                df = orderHistory_test_user.groupby('orderType').count()
                df.reset_index(inplace=True)
                for i in range(0, 2):
                    if (len(df[df['orderType'] == i]['orderTime'].values) != 0):
                        orderTypeX_num = df[df['orderType'] == i]['orderTime'].values[0]
                        orderTypeX_num_arr[i] += orderTypeX_num
    gc.collect()

    type_total_num = np.sum(typeX_num_arr)
    orderType_total_num = np.sum(orderTypeX_num_arr)
    for i in range(1, 10):
        future_time_file.write(',' + str(typeX_num_arr[i]))
    future_time_file.write(',' + str(type_total_num))
    for i in range(2):
        future_time_file.write(',' + str(orderTypeX_num_arr[i]))
    future_time_file.write(',' + str(orderType_total_num))
print('test耗时: ' + str(time.time() - t0))

# 特征情况二：按照行为事件从后往前排，时间越晚的行为权重越大，更改原来的orderTYpe_num
