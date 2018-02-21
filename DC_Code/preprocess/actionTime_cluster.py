#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-21 10:19:55
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : ${link}
# @Version : $Id$

import os
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import datetime
import time
import gc
from collections import defaultdict

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_path = project_path + "data/train/"
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
action_train = action_train[action_train['userid'] == 110106933065]
# action_train = action_train.head(2500)
t0 = time.time()
kequal1_num = 0
# file = open(project_path + 'preprocess/feature_file/actionTime_period_info_train.csv', 'w')
# file.write(
#     'userid,interval_mean,interval_lastvale1,interval_lastvale2,interval_lastvale3,next_actionTime,to_next_actionTime')
count = 0
userid_cluster_time = defaultdict(lambda: [])
for userid in action_train['userid'].unique():
    print(userid)
    count += 1
    if (count % 1000 == 0):
        print(count)
    # print('userid: ' + str(userid))
    # 按照历史订单情况划分簇
    user_orderHistory_arr = orderHistory_train['userid'].values
    actionTime_arr = action_train[action_train['userid'] == userid]['actionTime'].values
    print(len(actionTime_arr))
    if (userid in user_orderHistory_arr):
        orderTime_arr = orderHistory_train[orderHistory_train['userid'] == userid]['orderTime'].values
        if (orderTime_arr[0] >= actionTime_arr[-1] or orderTime_arr[-1] <= actionTime_arr[0]):
            k = 1
            arr1 = np.ones((len(actionTime_arr),))
            X = np.vstack((actionTime_arr, arr1)).T
            estimator = KMeans(n_clusters=k, random_state=0)
            y_pred = estimator.fit_predict(X)
            centroids = estimator.cluster_centers_  # 获取聚类中心
            cent_time_arr = centroids[:, 0]
            userid_cluster_time[userid].append(cent_time_arr[0])
        else:
            # print('else')
            time_arr_chonghe = np.array([])
            time_arr_chonghe = orderTime_arr[
                np.where((orderTime_arr > actionTime_arr[0]) & (orderTime_arr < actionTime_arr[-1]))]
            print('重合时间有： ' + str(len(time_arr_chonghe)) + '个')
            print(time_arr_chonghe)
            # 去除重复值以及actionTime首尾时间
            time_arr_chonghe_set = set(time_arr_chonghe.tolist())
            if (actionTime_arr[0] in time_arr_chonghe_set):
                time_arr_chonghe_set = time_arr_chonghe_set - set([actionTime_arr[0]])
            elif (actionTime_arr[-1] in time_arr_chonghe_set):
                time_arr_chonghe_set = time_arr_chonghe_set - set([actionTime_arr[-1]])
            print('set len: ')
            print(len(time_arr_chonghe_set))
            time_arr_chonghe = np.array(sorted(list(time_arr_chonghe_set)))
            print('去重后重合时间有： ' + str(len(time_arr_chonghe)) + '个')
            print(str(userid) + '按照orderTime 和 actionTime分成 ' + str(len(time_arr_chonghe) + 1) + ' 个簇 ')
            # print(actionTime_arr.tolist())
            actionTime_list = actionTime_arr.tolist()
            print(len(actionTime_list))
            time_chonghe_list = time_arr_chonghe.tolist()
            print(len(time_chonghe_list))
            actionTime_list.extend(time_chonghe_list)
            print(len(actionTime_list))
            # time_list = actionTime_arr.tolist().extend(time_arr_chonghe.tolist())
            # print(time_list)
            all_time_arr = sorted(np.array(actionTime_list))
            index_a = 0
            index_b = 0
            k = 1
            print(len(all_time_arr))
            for i in range(len(time_arr_chonghe) + 1):
                print('i= ' + str(i))
                # np.argwhere(b == 1)[0][0]
                if (i != len(time_arr_chonghe)):
                    print(time_arr_chonghe[i])
                index_b = np.argwhere(all_time_arr == time_arr_chonghe[i])[0][0] if (i != len(time_arr_chonghe)) else -1
                print('index_a: ' + str(index_a))
                print('index_b: ' + str(index_b))
                if (index_b == -1):
                    action_time_arr = all_time_arr[index_a:]
                else:
                    action_time_arr = all_time_arr[index_a:index_b]
                print(len(action_time_arr))
                index_a = index_b
                arr1 = np.ones((len(action_time_arr),))
                X = np.vstack((action_time_arr, arr1)).T
                estimator = KMeans(n_clusters=k, random_state=0)
                y_pred = estimator.fit_predict(X)
                centroids = estimator.cluster_centers_  # 获取聚类中心
                cent_time_arr = centroids[:, 0]
                userid_cluster_time[userid].append(round(cent_time_arr[0], 2))
    else:
        k = 1
        arr1 = np.ones((len(actionTime_arr),))
        X = np.vstack((actionTime_arr, arr1)).T
        estimator = KMeans(n_clusters=k, random_state=0)
        y_pred = estimator.fit_predict(X)
        centroids = estimator.cluster_centers_  # 获取聚类中心
        cent_time_arr = centroids[:, 0]
        userid_cluster_time[userid].append(cent_time_arr[0])
    gc.collect()

# print(kequal1_num)
print('train花费时间： ' + str(time.time() - t0))
print(userid_cluster_time)


# userid_duliang_useridother = defaultdict(defaultdict(lambda: 0))
# for userid in userid_cluster_time.keys():
#     for userid_other in (set(userid_cluster_time.keys()) - set(userid)):
#         for i in range(len(userid_cluster_time[userid_other])):


# testdata_path = project_path + "data/test/"
# action_test = pd.read_csv(testdata_path + "action_test.csv")
# # action_test = action_test[action_test['userid'] == 100000000013]
# # action_test = action_test.head(100)
# t0 = time.time()
# kequal1_num = 0
# file = open(project_path + 'preprocess/feature_file/actionTime_period_info_test.csv', 'w')
# file.write(
#     'userid,interval_mean,interval_lastvale1,interval_lastvale2,interval_lastvale3,next_actionTime,to_next_actionTime')
# count = 0
# for userid in action_test['userid'].unique():
#     count += 1
#     if (count % 1000 == 0):
#         print(count)
#     # print('userid: ' + str(userid))
#     actionTime = action_test[action_test['userid'] == userid]['actionTime'].values
#     # 寻找k值，actionTime间隔超过3600s，即一个小时，k+1
#     k = 1  # 至少一个簇
#     for i in range(1, len(actionTime)):
#         if (actionTime[i] - actionTime[i - 1] > 3600):
#             k += 1
#     # print('簇个数： ' + str(k))
#     arr1 = np.ones((len(actionTime),))
#     X = np.vstack((actionTime, arr1)).T
#     estimator = KMeans(n_clusters=k, random_state=0)
#     y_pred = estimator.fit_predict(X)
#     centroids = estimator.cluster_centers_  # 获取聚类中心
#     cent_time_arr = centroids[:, 0]
#     cent_time_arr = sorted(cent_time_arr)
#     if (k == 1):
#         kequal1_num += 1
#     # 现在行为周期时间戳保存在cent_time_arr里，我可以得到行为周期间隔的均值，以及下次行为时间。
#     action_period_interval_list = []
#     for i in range(1, len(cent_time_arr)):
#         action_period_interval_list.append(cent_time_arr[i] - cent_time_arr[i - 1])
#     # print('actionTime_period_interval_len: ' + str(len(action_period_interval_list)))
#     if (len(action_period_interval_list) != 0):
#         interval_mean = round(np.mean(np.array(action_period_interval_list)), 2)
#         interval_lastvale1 = action_period_interval_list[-1]
#         interval_lastvale2 = action_period_interval_list[-2] if (len(action_period_interval_list) > 1) else np.nan
#         interval_lastvale3 = action_period_interval_list[-3] if (len(action_period_interval_list) > 2) else np.nan
#         if (np.isnan(interval_lastvale2) or np.isnan(interval_lastvale3)):
#             # print('userid')
#             # break
#             next_actionTime = cent_time_arr[-1] + interval_mean
#             to_next_actionTime = next_actionTime - actionTime[-1]
#         else:
#             next_actionTime = cent_time_arr[-1] + round(
#                 np.mean([interval_lastvale1, interval_lastvale2, interval_lastvale3]), 2)
#             to_next_actionTime = next_actionTime - actionTime[-1]
#     else:
#         interval_mean = np.nan
#         interval_lastvale1 = np.nan
#         interval_lastvale2 = np.nan
#         interval_lastvale3 = np.nan
#         next_actionTime = np.nan
#         to_next_actionTime = np.nan
#     file.write('\n')
#     file.write(str(userid) + ',' + str(interval_mean) + ',' + str(interval_lastvale1) + ',' + str(
#         interval_lastvale2) + ',' + str(
#         interval_lastvale3) + ',' + str(next_actionTime) + ',' + str(to_next_actionTime))
#     gc.collect()
#
# # print(kequal1_num)
# print('test花费时间： ' + str(time.time() - t0))
