#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-02 20:16:22
# @Author  : guanglinzhou (xdzgl812@163.com)

# 通过对比orderHistory_train的orderTime和action_train的actionTime，将orderHistory_train和action_train合并到一起

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

################
project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'
traindata_path = project_path + "data/train/"
traindata_output_path = project_path + "preprocess/"
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderFuture_train = pd.read_csv(traindata_path + "orderFuture_train.csv")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
userComment_train = pd.read_csv(traindata_path + "userComment_train.csv")
userProfile_train = pd.read_csv(traindata_path + "userProfile_train.csv")
# train
history_action_merge_by_time_train = open(
    project_path + 'preprocess/middle_file/history_action_merge_by_time_train.csv', 'w')
history_action_merge_by_time_train.write(
    'userid,actionType,time,actionTime_zero_orderTime_one,orderType,orderid,city,country,continent')
action_train_userid_set = set(action_train['userid'])
orderHistory_train_userid_set = set(orderHistory_train['userid'])
print('\ntrain...\n')
print('共有' + str(len(action_train_userid_set)) + '个userid\n')
num = 0
for userid in action_train_userid_set:
    num += 1
    if (num % 1000 == 0):
        print('已经处理了' + str(num) + '个user\n')
    if (userid in orderHistory_train['userid'].values):
        orderTime_list = orderHistory_train[orderHistory_train['userid'] == userid]['orderTime'].values.tolist()
        # print(orderTime_list)
        actionTime_list = action_train[action_train['userid'] == userid]['actionTime'].values.tolist()
        # print(actionTime_list)
        time_list = []
        time_list.extend(orderTime_list)
        time_list.extend(actionTime_list)
        time_list.sort()
        for time in time_list:
            if (time in orderTime_list):
                write_userid = str(userid)
                write_actionType = ''
                write_time = str(time)
                write_actionTime_zero_orderTime_one = str(1)
                write_orderType = str(orderHistory_train[(orderHistory_train['userid'] == userid) & (
                        orderHistory_train['orderTime'] == time)]['orderType'].values[0])
                write_orderid = str(orderHistory_train[(orderHistory_train['userid'] == userid) & (
                        orderHistory_train['orderTime'] == time)]['orderid'].values[0])
                write_city = str(orderHistory_train[(orderHistory_train['userid'] == userid) & (
                        orderHistory_train['orderTime'] == time)]['city'].values[0])
                write_country = str(orderHistory_train[(orderHistory_train['userid'] == userid) & (
                        orderHistory_train['orderTime'] == time)]['country'].values[0])
                write_continent = str(orderHistory_train[(orderHistory_train['userid'] == userid) & (
                        orderHistory_train['orderTime'] == time)]['continent'].values[0])

                history_action_merge_by_time_train.write('\n')
                history_action_merge_by_time_train.write(
                    write_userid + ',' + write_actionType + ',' + write_time + ',' + write_actionTime_zero_orderTime_one + ',' + write_orderType + ',' + write_orderid + ',' + write_city + ',' + write_country + ',' + write_continent)
            else:
                write_userid = str(userid)
                write_time = str(time)
                write_actionType = str(
                    action_train[(action_train['userid'] == userid) & (action_train['actionTime'] == int(write_time))][
                        'actionType'].values[0])
                write_actionTime_zero_orderTime_one = str(0)
                write_orderType = ''
                write_orderid = ''
                write_city = ''
                write_country = ''
                write_continent = ''

                history_action_merge_by_time_train.write('\n')
                history_action_merge_by_time_train.write(
                    write_userid + ',' + write_actionType + ',' + write_time + ',' + write_actionTime_zero_orderTime_one + ',' + write_orderType + ',' + write_orderid + ',' + write_city + ',' + write_country + ',' + write_continent)
    else:
        write_userid = str(userid)
        write_time = str(action_train[action_train['userid'] == userid]['actionTime'].values[0])
        write_actionType = str((action_train[
            (action_train['userid'] == userid) & (action_train['actionTime'] == int(write_time))]['actionType'].values[
            0]))
        write_actionTime_zero_orderTime_one = str(0)
        write_orderType = ''
        write_orderid = ''
        write_city = ''
        write_country = ''
        write_continent = ''

        history_action_merge_by_time_train.write('\n')
        history_action_merge_by_time_train.write(
            write_userid + ',' + write_actionType + ',' + write_time + ',' + write_actionTime_zero_orderTime_one + ',' + write_orderType + ',' + write_orderid + ',' + write_city + ',' + write_country + ',' + write_continent)
history_action_merge_by_time_train.close()

# test
testdata_path = project_path + "data/test/"
testdata_output_path = project_path + "preprocess/"
action_test = pd.read_csv(testdata_path + "action_test.csv")
orderFuture_test = pd.read_csv(testdata_path + "orderFuture_test.csv")
orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")
userComment_test = pd.read_csv(testdata_path + "userComment_test.csv")
userProfile_test = pd.read_csv(testdata_path + "userProfile_test.csv")

history_action_merge_by_time_test = open(
    project_path + 'preprocess/middle_file/history_action_merge_by_time_test.csv', 'w')
history_action_merge_by_time_test.write(
    'userid,actionType,time,actionTime_zero_orderTime_one,orderType,orderid,city,country,continent')
action_test_userid_set = set(action_test['userid'])
orderHistory_test_userid_set = set(orderHistory_test['userid'])
print('\ntest...\n')
print('共有' + str(len(action_test_userid_set)) + '个userid\n')
num = 0
for userid in action_test_userid_set:
    num += 1
    if (num % 1000 == 0):
        print('已经处理了' + str(num) + '个user\n')
    if (userid in orderHistory_test['userid'].values):
        orderTime_list = orderHistory_test[orderHistory_test['userid'] == userid]['orderTime'].values.tolist()
        # print(orderTime_list)
        actionTime_list = action_test[action_test['userid'] == userid]['actionTime'].values.tolist()
        # print(actionTime_list)
        time_list = []
        time_list.extend(orderTime_list)
        time_list.extend(actionTime_list)
        time_list.sort()
        for time in time_list:
            if (time in orderTime_list):
                write_userid = str(userid)
                write_actionType = ''
                write_time = str(time)
                write_actionTime_zero_orderTime_one = str(1)
                write_orderType = str(orderHistory_test[(orderHistory_test['userid'] == userid) & (
                        orderHistory_test['orderTime'] == time)]['orderType'].values[0])
                write_orderid = str(orderHistory_test[(orderHistory_test['userid'] == userid) & (
                        orderHistory_test['orderTime'] == time)]['orderid'].values[0])
                write_city = str(orderHistory_test[(orderHistory_test['userid'] == userid) & (
                        orderHistory_test['orderTime'] == time)]['city'].values[0])
                write_country = str(orderHistory_test[(orderHistory_test['userid'] == userid) & (
                        orderHistory_test['orderTime'] == time)]['country'].values[0])
                write_continent = str(orderHistory_test[(orderHistory_test['userid'] == userid) & (
                        orderHistory_test['orderTime'] == time)]['continent'].values[0])

                history_action_merge_by_time_test.write('\n')
                history_action_merge_by_time_test.write(
                    write_userid + ',' + write_actionType + ',' + write_time + ',' + write_actionTime_zero_orderTime_one + ',' + write_orderType + ',' + write_orderid + ',' + write_city + ',' + write_country + ',' + write_continent)
            else:
                write_userid = str(userid)
                write_time = str(time)
                write_actionType = str(
                    action_test[(action_test['userid'] == userid) & (action_test['actionTime'] == int(write_time))][
                        'actionType'].values[0])
                write_actionTime_zero_orderTime_one = str(0)
                write_orderType = ''
                write_orderid = ''
                write_city = ''
                write_country = ''
                write_continent = ''

                history_action_merge_by_time_test.write('\n')
                history_action_merge_by_time_test.write(
                    write_userid + ',' + write_actionType + ',' + write_time + ',' + write_actionTime_zero_orderTime_one + ',' + write_orderType + ',' + write_orderid + ',' + write_city + ',' + write_country + ',' + write_continent)
    else:
        write_userid = str(userid)
        write_time = str(action_test[action_test['userid'] == userid]['actionTime'].values[0])
        write_actionType = str((action_test[
            (action_test['userid'] == userid) & (action_test['actionTime'] == int(write_time))]['actionType'].values[
            0]))
        write_actionTime_zero_orderTime_one = str(0)
        write_orderType = ''
        write_orderid = ''
        write_city = ''
        write_country = ''
        write_continent = ''

        history_action_merge_by_time_test.write('\n')
        history_action_merge_by_time_test.write(
            write_userid + ',' + write_actionType + ',' + write_time + ',' + write_actionTime_zero_orderTime_one + ',' + write_orderType + ',' + write_orderid + ',' + write_city + ',' + write_country + ',' + write_continent)
history_action_merge_by_time_test.close()
