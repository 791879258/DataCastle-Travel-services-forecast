#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-29 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn import preprocessing
import gc

# 特征均可通过merge操作，on='userid'来合并到表xxxdata_output.csv中

###########-----------------------读表----------------------------###########
project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'
traindata_path = project_path + "data/train/"
traindata_path = project_path + "data/train/"
testdata_path = project_path + "data/test/"
feature_file_path = project_path + 'preprocess/feature_file/feature_from_jingsaiquan/'

# 竞赛圈的统计特征

# # 用户行为数据中倒数前三个Type是多少,第一个type是多少
# action_train = pd.read_csv(traindata_path + "action_train.csv")
# action_test = pd.read_csv(testdata_path + "action_test.csv")
#
#
# def fun_daoshuType_123(df):
#     global num_daoshuType_123
#     actionType_arr = df['actionType'].values
#     if (num_daoshuType_123 != 0):
#         userid_daoshuType_123_dict[df['userid'].values[0]].append(actionType_arr[0])
#         userid_daoshuType_123_dict[df['userid'].values[0]].append(actionType_arr[-1])
#         if (len(actionType_arr) == 1):
#             userid_daoshuType_123_dict[df['userid'].values[0]].append(np.nan)
#             userid_daoshuType_123_dict[df['userid'].values[0]].append(np.nan)
#         else:
#             userid_daoshuType_123_dict[df['userid'].values[0]].append(int(actionType_arr[-2]))
#             if (len(actionType_arr) == 2):
#                 userid_daoshuType_123_dict[df['userid'].values[0]].append(np.nan)
#             else:
#                 userid_daoshuType_123_dict[df['userid'].values[0]].append(int(actionType_arr[-3]))
#     num_daoshuType_123 += 1
#     return df
#
#
# num_daoshuType_123 = 0
# userid_daoshuType_123_dict = defaultdict(lambda: [])
# action_train.groupby('userid').apply(fun_daoshuType_123)
# daoshuType_123 = pd.DataFrame.from_dict(userid_daoshuType_123_dict, orient='index').reset_index()
# daoshuType_123.rename(columns={'index': 'userid', 0: 'firstType', 1: 'daoshuType1', 2: 'daoshuType2', 3: 'daoshuType3'},
#                       inplace=True)
# daoshuType_123.to_csv(feature_file_path + 'daoshuType_123_train.csv', index=False)
#
# num_daoshuType_123 = 0
# userid_daoshuType_123_dict = defaultdict(lambda: [])
# action_test.groupby('userid').apply(fun_daoshuType_123)
# daoshuType_123 = pd.DataFrame.from_dict(userid_daoshuType_123_dict, orient='index').reset_index()
# daoshuType_123.rename(columns={'index': 'userid', 0: 'firstType', 1: 'daoshuType1', 2: 'daoshuType2', 3: 'daoshuType3'},
#                       inplace=True)
# daoshuType_123.to_csv(feature_file_path + 'daoshuType_123_test.csv', index=False)

# # 用户行为数据中时间间隔的统计信息，时间间隔的均值，方差，最小值，末尾值，时间间隔倒数第二、三、四个值，最后三个时间间隔均值、方差，第一个时间间隔。
# action_train = pd.read_csv(traindata_path + "action_train.csv")
# action_test = pd.read_csv(testdata_path + "action_test.csv")
#
#
# def fun_time_interval(df):
#     global num_time_interval
#     time_interval = []
#     actionTime_arr = df['actionTime'].values
#     if (num_time_interval != 0):
#         for i in range(len(actionTime_arr)):
#             if ((i + 1) == len(actionTime_arr)):
#                 break
#             else:
#                 time_interval.append(actionTime_arr[i + 1] - actionTime_arr[i])
#
#         userid_time_interval_dict[df['userid'].values[0]].extend(time_interval)
#     num_time_interval += 1
#     return df
#
#
# num_time_interval = 0
# # {userid:[interval1,interval2,...]}
# userid_time_interval_dict = defaultdict(lambda: [])
# action_train.groupby('userid').apply(fun_time_interval)
#
# # time_interval_feature_list = []
# userid_time_interval_feature_dict_train = defaultdict(lambda: [])
# num_cnt = 0
# print(len(userid_time_interval_dict.keys()))
# for userid in userid_time_interval_dict.keys():
#     # for userid in list(userid_time_interval_dict.keys())[:1000]:
#
#     # for userid in [100000000013]:
#     num_cnt += 1
#     if (num_cnt % 100 == 0):
#         print(num_cnt)
#     userid_time_interval_array = np.asarray(userid_time_interval_dict[userid])
#     if (len(userid_time_interval_array) == 0):
#         interval_mean = np.nan
#         interval_var = np.nan
#         interval_min = np.nan
#         interval_lastvale1 = np.nan
#         interval_lastvale2 = np.nan
#         interval_lastvale3 = np.nan
#         interval_lastvale4 = np.nan
#         interval_last3_mean = np.nan
#         interval_last3_var = np.nan
#         first_interval = np.nan
#     else:
#         interval_mean = round(np.mean(userid_time_interval_array), 2) if (
#                 len(userid_time_interval_array) > 1) else np.nan
#         interval_var = round(np.var(userid_time_interval_array), 2) if (
#                 len(userid_time_interval_array) > 1) else np.nan
#         interval_min = np.min(userid_time_interval_array)
#         interval_lastvale1 = userid_time_interval_array[-1]
#         interval_lastvale2 = userid_time_interval_array[-2] if (len(userid_time_interval_array) > 1) else np.nan
#         interval_lastvale3 = userid_time_interval_array[-3] if (len(userid_time_interval_array) > 2) else np.nan
#         interval_lastvale4 = userid_time_interval_array[-4] if (len(userid_time_interval_array) > 3) else np.nan
#         interval_last3_arr = np.asarray([interval_lastvale1, interval_lastvale2, interval_lastvale3])
#         interval_last3_mean = round(np.mean(interval_last3_arr), 2) if (len(userid_time_interval_array) > 2) else np.nan
#         interval_last3_var = round(np.var(interval_last3_arr), 2) if (len(userid_time_interval_array) > 2) else np.nan
#         first_interval = userid_time_interval_array[0]
#
#     userid_time_interval_feature_dict_train[userid].append(interval_mean)
#     userid_time_interval_feature_dict_train[userid].append(interval_var)
#     userid_time_interval_feature_dict_train[userid].append(interval_min)
#     userid_time_interval_feature_dict_train[userid].append(interval_lastvale1)
#     userid_time_interval_feature_dict_train[userid].append(interval_lastvale2)
#     userid_time_interval_feature_dict_train[userid].append(interval_lastvale3)
#     userid_time_interval_feature_dict_train[userid].append(interval_lastvale4)
#     userid_time_interval_feature_dict_train[userid].append(interval_last3_mean)
#     userid_time_interval_feature_dict_train[userid].append(interval_last3_var)
#     userid_time_interval_feature_dict_train[userid].append(first_interval)
#     # print(len(userid_time_interval_feature_dict_train[userid]))
#     gc.collect()
# # print(userid_time_interval_feature_dict_train)
# userid_time_interval_df_train = pd.DataFrame.from_dict(userid_time_interval_feature_dict_train,
#                                                        orient='index').reset_index()
# # print(userid_time_interval_df_train)
# userid_time_interval_df_train.rename(columns={'index': 'userid', 0:
#     'interval_mean', 1: 'interval_var', 2: 'interval_min', 3: 'interval_lastvale1', 4: 'interval_lastvale2',
#                                               5: 'interval_lastvale3', 6: 'interval_lastvale4',
#                                               7: 'interval_last3_mean', 8: 'interval_last3_var', 9: 'first_interval'},
#                                      inplace=True)
# userid_time_interval_df_train.to_csv(feature_file_path + 'userid_time_interval_train.csv', index=False)
#
# num_time_interval = 0
# # {userid:[interval1,interval2,...]}
# userid_time_interval_dict = defaultdict(lambda: [])
# action_test.groupby('userid').apply(fun_time_interval)
#
# # time_interval_feature_list = []
# userid_time_interval_feature_dict_test = defaultdict(lambda: [])
# num_cnt = 0
# # print(len(userid_time_interval_dict.keys()))
# for userid in userid_time_interval_dict.keys():
#     # for userid in list(userid_time_interval_dict.keys())[:1000]:
#
#     num_cnt += 1
#     if (num_cnt % 100 == 0):
#         print(num_cnt)
#     userid_time_interval_array = np.asarray(userid_time_interval_dict[userid])
#     if (len(userid_time_interval_array) == 0):
#         interval_mean = np.nan
#         interval_var = np.nan
#         interval_min = np.nan
#         interval_lastvale1 = np.nan
#         interval_lastvale2 = np.nan
#         interval_lastvale3 = np.nan
#         interval_lastvale4 = np.nan
#         interval_last3_mean = np.nan
#         interval_last3_var = np.nan
#         first_interval = np.nan
#     else:
#         interval_mean = round(np.mean(userid_time_interval_array), 2) if (
#                 len(userid_time_interval_array) > 1) else np.nan
#         interval_var = round(np.var(userid_time_interval_array), 2) if (
#                 len(userid_time_interval_array) > 1) else np.nan
#         interval_min = np.min(userid_time_interval_array)
#         interval_lastvale1 = userid_time_interval_array[-1]
#         interval_lastvale2 = userid_time_interval_array[-2] if (len(userid_time_interval_array) > 1) else np.nan
#         interval_lastvale3 = userid_time_interval_array[-3] if (len(userid_time_interval_array) > 2) else np.nan
#         interval_lastvale4 = userid_time_interval_array[-4] if (len(userid_time_interval_array) > 3) else np.nan
#         interval_last3_arr = np.asarray([interval_lastvale1, interval_lastvale2, interval_lastvale3])
#         interval_last3_mean = round(np.mean(interval_last3_arr), 2) if (len(userid_time_interval_array) > 2) else np.nan
#         interval_last3_var = round(np.var(interval_last3_arr), 2) if (len(userid_time_interval_array) > 2) else np.nan
#         first_interval = userid_time_interval_array[0]
#
#     userid_time_interval_feature_dict_test[userid].append(interval_mean)
#     userid_time_interval_feature_dict_test[userid].append(interval_var)
#     userid_time_interval_feature_dict_test[userid].append(interval_min)
#     userid_time_interval_feature_dict_test[userid].append(interval_lastvale1)
#     userid_time_interval_feature_dict_test[userid].append(interval_lastvale2)
#     userid_time_interval_feature_dict_test[userid].append(interval_lastvale3)
#     userid_time_interval_feature_dict_test[userid].append(interval_lastvale4)
#     userid_time_interval_feature_dict_test[userid].append(interval_last3_mean)
#     userid_time_interval_feature_dict_test[userid].append(interval_last3_var)
#     userid_time_interval_feature_dict_test[userid].append(first_interval)
#     gc.collect()
#
# userid_time_interval_df_test = pd.DataFrame.from_dict(userid_time_interval_feature_dict_test,
#                                                       orient='index').reset_index()
# userid_time_interval_df_test.rename(columns={'index': 'userid', 0:
#     'interval_mean', 1: 'interval_var', 2: 'interval_min', 3: 'interval_lastvale1', 4: 'interval_lastvale2',
#                                              5: 'interval_lastvale3', 6: 'interval_lastvale4',
#                                              7: 'interval_last3_mean', 8: 'interval_last3_var', 9: 'first_interval'},
#                                     inplace=True)
# userid_time_interval_df_test.to_csv(feature_file_path + 'userid_time_interval_test.csv', index=False)

# # 年龄和省份
# userProfile_train = pd.read_csv(traindata_path + 'userProfile_train.csv')
# userProfile_test = pd.read_csv(testdata_path + 'userProfile_test.csv')
# df = pd.get_dummies(userProfile_train, columns=['age'])
# df = pd.get_dummies(df, columns=['province'])
# del df['gender']
# df.to_csv(feature_file_path + 'province_age_train.csv', index=False)
#
# df = pd.get_dummies(userProfile_test, columns=['age'])
# df = pd.get_dummies(df, columns=['province'])
# del df['gender']
# df.to_csv(feature_file_path + 'province_age_test.csv', index=False)


# # 用户行为数据中离最近的TypeX的距离和时间
# action_train = pd.read_csv(traindata_path + "action_train.csv")
# action_test = pd.read_csv(testdata_path + "action_test.csv")
#
#
# # 直接在groupby(func)的func中写入表格，两个一维数组，一个表示Type，另一个表示Type对应的时间
#
# def fun_to_closest_X_DistandTime(df):
#     global num_to_closest_X_DistandTime
#     global num_count
#     num_count += 1
#     if (num_count % 1000 == 0):
#         print(str(num_count) + '  done...')
#     if (num_to_closest_X_DistandTime != 0):
#         actionType_arr = df['actionType'].values
#         actionTime_arr = df['actionTime'].values
#         arr_length = len(actionTime_arr)
#         if (1 in actionType_arr):
#             index_1 = np.where(actionType_arr == 1)[0]
#             closest_1_time = actionTime_arr[index_1[-1]]
#             to_closest_1_time = actionTime_arr[-1] - actionTime_arr[index_1[-1]]
#             to_closest_1_dist = arr_length - 1 - index_1[-1]
#             to_closest_1_time_interval_list = []
#             if (index_1[-1] == arr_length - 1):
#                 to_closest_1_time_interval_mean = np.nan
#                 to_closest_1_time_interval_var = np.nan
#                 to_closest_1_time_interval_min = np.nan
#                 to_closest_1_time_interval_max = np.nan
#                 to_closest_1_time_interval_median = np.nan
#             else:
#                 for i in range(index_1[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_1_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_1_time_interval_arr = np.asarray(to_closest_1_time_interval_list)
#                 to_closest_1_time_interval_mean = round(np.mean(to_closest_1_time_interval_arr), 2) if (
#                         len(to_closest_1_time_interval_arr) > 1) else np.nan
#                 to_closest_1_time_interval_var = round(np.var(to_closest_1_time_interval_arr), 2) if (
#                         len(to_closest_1_time_interval_arr) > 1) else np.nan
#                 to_closest_1_time_interval_min = np.min(to_closest_1_time_interval_arr)
#                 to_closest_1_time_interval_max = np.max(to_closest_1_time_interval_arr)
#                 to_closest_1_time_interval_median = np.median(to_closest_1_time_interval_arr)
#         else:
#             closest_1_time = np.nan
#             to_closest_1_dist = np.nan
#             to_closest_1_time_interval_mean = np.nan
#             to_closest_1_time_interval_var = np.nan
#             to_closest_1_time_interval_min = np.nan
#             to_closest_1_time_interval_max = np.nan
#             to_closest_1_time_interval_median = np.nan
#             to_closest_1_time = np.nan
#
#         if (2 in actionType_arr):
#             index_2 = np.where(actionType_arr == 2)[0]
#             closest_2_time = actionTime_arr[index_2[-1]]
#             to_closest_2_time = actionTime_arr[-1] - actionTime_arr[index_2[-1]]
#             to_closest_2_dist = arr_length - 1 - index_2[-1]
#             to_closest_2_time_interval_list = []
#             if (index_2[-1] == arr_length - 1):
#                 to_closest_2_time_interval_mean = np.nan
#                 to_closest_2_time_interval_var = np.nan
#                 to_closest_2_time_interval_min = np.nan
#                 to_closest_2_time_interval_max = np.nan
#                 to_closest_2_time_interval_median = np.nan
#             else:
#                 for i in range(index_2[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_2_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_2_time_interval_arr = np.asarray(to_closest_2_time_interval_list)
#                 to_closest_2_time_interval_mean = round(np.mean(to_closest_2_time_interval_arr), 2) if (
#                         len(to_closest_2_time_interval_arr) > 1) else np.nan
#                 to_closest_2_time_interval_var = round(np.var(to_closest_2_time_interval_arr), 2) if (
#                         len(to_closest_2_time_interval_arr) > 1) else np.nan
#                 to_closest_2_time_interval_min = np.min(to_closest_2_time_interval_arr)
#                 to_closest_2_time_interval_max = np.max(to_closest_2_time_interval_arr)
#                 to_closest_2_time_interval_median = np.median(to_closest_2_time_interval_arr)
#         else:
#             closest_2_time = np.nan
#             to_closest_2_time = np.nan
#             to_closest_2_dist = np.nan
#             to_closest_2_time_interval_mean = np.nan
#             to_closest_2_time_interval_var = np.nan
#             to_closest_2_time_interval_min = np.nan
#             to_closest_2_time_interval_max = np.nan
#             to_closest_2_time_interval_median = np.nan
#
#         if (3 in actionType_arr):
#             index_3 = np.where(actionType_arr == 3)[0]
#             closest_3_time = actionTime_arr[index_3[-1]]
#             to_closest_3_time = actionTime_arr[-1] - actionTime_arr[index_3[-1]]
#             to_closest_3_dist = arr_length - 1 - index_3[-1]
#             to_closest_3_time_interval_list = []
#             if (index_3[-1] == arr_length - 1):
#                 to_closest_3_time_interval_mean = np.nan
#                 to_closest_3_time_interval_var = np.nan
#                 to_closest_3_time_interval_min = np.nan
#                 to_closest_3_time_interval_max = np.nan
#                 to_closest_3_time_interval_median = np.nan
#             else:
#                 for i in range(index_3[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_3_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_3_time_interval_arr = np.asarray(to_closest_3_time_interval_list)
#                 to_closest_3_time_interval_mean = round(np.mean(to_closest_3_time_interval_arr), 2) if (
#                         len(to_closest_3_time_interval_arr) > 1) else np.nan
#                 to_closest_3_time_interval_var = round(np.var(to_closest_3_time_interval_arr), 2) if (
#                         len(to_closest_3_time_interval_arr) > 1) else np.nan
#                 to_closest_3_time_interval_min = np.min(to_closest_3_time_interval_arr)
#                 to_closest_3_time_interval_max = np.max(to_closest_3_time_interval_arr)
#                 to_closest_3_time_interval_median = np.median(to_closest_3_time_interval_arr)
#         else:
#             closest_3_time = np.nan
#             to_closest_3_time = np.nan
#             to_closest_3_dist = np.nan
#             to_closest_3_time_interval_mean = np.nan
#             to_closest_3_time_interval_var = np.nan
#             to_closest_3_time_interval_min = np.nan
#             to_closest_3_time_interval_max = np.nan
#             to_closest_3_time_interval_median = np.nan
#
#         if (4 in actionType_arr):
#             index_4 = np.where(actionType_arr == 4)[0]
#             closest_4_time = actionTime_arr[index_4[-1]]
#             to_closest_4_time = actionTime_arr[-1] - actionTime_arr[index_4[-1]]
#             to_closest_4_dist = arr_length - 1 - index_4[-1]
#             to_closest_4_time_interval_list = []
#             if (index_4[-1] == arr_length - 1):
#                 to_closest_4_time_interval_mean = np.nan
#                 to_closest_4_time_interval_var = np.nan
#                 to_closest_4_time_interval_min = np.nan
#                 to_closest_4_time_interval_max = np.nan
#                 to_closest_4_time_interval_median = np.nan
#             else:
#                 for i in range(index_4[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_4_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_4_time_interval_arr = np.asarray(to_closest_4_time_interval_list)
#                 to_closest_4_time_interval_mean = round(np.mean(to_closest_4_time_interval_arr), 2) if (
#                         len(to_closest_4_time_interval_arr) > 1) else np.nan
#                 to_closest_4_time_interval_var = round(np.var(to_closest_4_time_interval_arr), 2) if (
#                         len(to_closest_4_time_interval_arr) > 1) else np.nan
#                 to_closest_4_time_interval_min = np.min(to_closest_4_time_interval_arr)
#                 to_closest_4_time_interval_max = np.max(to_closest_4_time_interval_arr)
#                 to_closest_4_time_interval_median = np.median(to_closest_4_time_interval_arr)
#         else:
#             closest_4_time = np.nan
#             to_closest_4_time = np.nan
#             to_closest_4_dist = np.nan
#             to_closest_4_time_interval_mean = np.nan
#             to_closest_4_time_interval_var = np.nan
#             to_closest_4_time_interval_min = np.nan
#             to_closest_4_time_interval_max = np.nan
#             to_closest_4_time_interval_median = np.nan
#         if (5 in actionType_arr):
#             index_5 = np.where(actionType_arr == 5)[0]
#             closest_5_time = actionTime_arr[index_5[-1]]
#             to_closest_5_time = actionTime_arr[-1] - actionTime_arr[index_5[-1]]
#             to_closest_5_dist = arr_length - 1 - index_5[-1]
#             to_closest_5_time_interval_list = []
#             if (index_5[-1] == arr_length - 1):
#                 to_closest_5_time_interval_mean = np.nan
#                 to_closest_5_time_interval_var = np.nan
#                 to_closest_5_time_interval_min = np.nan
#                 to_closest_5_time_interval_max = np.nan
#                 to_closest_5_time_interval_median = np.nan
#             else:
#                 for i in range(index_5[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_5_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_5_time_interval_arr = np.asarray(to_closest_5_time_interval_list)
#                 to_closest_5_time_interval_mean = round(np.mean(to_closest_5_time_interval_arr), 2) if (
#                         len(to_closest_5_time_interval_arr) > 1) else np.nan
#                 to_closest_5_time_interval_min = np.min(to_closest_5_time_interval_arr)
#                 to_closest_5_time_interval_var = round(np.var(to_closest_5_time_interval_arr), 2) if (
#                         len(to_closest_5_time_interval_arr) > 1) else np.nan
#                 to_closest_5_time_interval_max = np.max(to_closest_5_time_interval_arr)
#                 to_closest_5_time_interval_median = np.median(to_closest_5_time_interval_arr)
#         else:
#             closest_5_time = np.nan
#             to_closest_5_time = np.nan
#             to_closest_5_dist = np.nan
#             to_closest_5_time_interval_mean = np.nan
#             to_closest_5_time_interval_var = np.nan
#             to_closest_5_time_interval_min = np.nan
#             to_closest_5_time_interval_max = np.nan
#             to_closest_5_time_interval_median = np.nan
#
#         if (6 in actionType_arr):
#             index_6 = np.where(actionType_arr == 6)[0]
#             closest_6_time = actionTime_arr[index_6[-1]]
#             to_closest_6_time = actionTime_arr[-1] - actionTime_arr[index_6[-1]]
#             to_closest_6_dist = arr_length - 1 - index_6[-1]
#             to_closest_6_time_interval_list = []
#             if (index_6[-1] == arr_length - 1):
#                 to_closest_6_time_interval_mean = np.nan
#                 to_closest_6_time_interval_var = np.nan
#                 to_closest_6_time_interval_min = np.nan
#                 to_closest_6_time_interval_max = np.nan
#                 to_closest_6_time_interval_median = np.nan
#
#             else:
#                 for i in range(index_6[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_6_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_6_time_interval_arr = np.asarray(to_closest_6_time_interval_list)
#                 to_closest_6_time_interval_mean = round(np.mean(to_closest_6_time_interval_arr), 2) if (
#                         len(to_closest_6_time_interval_arr) > 1) else np.nan
#                 to_closest_6_time_interval_min = np.min(to_closest_6_time_interval_arr)
#                 to_closest_6_time_interval_var = round(np.var(to_closest_6_time_interval_arr), 2) if (
#                         len(to_closest_6_time_interval_arr) > 1) else np.nan
#                 to_closest_6_time_interval_max = np.max(to_closest_6_time_interval_arr)
#                 to_closest_6_time_interval_median = np.median(to_closest_6_time_interval_arr)
#         else:
#             closest_6_time = np.nan
#             to_closest_6_time = np.nan
#             to_closest_6_dist = np.nan
#             to_closest_6_time_interval_mean = np.nan
#             to_closest_6_time_interval_var = np.nan
#             to_closest_6_time_interval_min = np.nan
#             to_closest_6_time_interval_max = np.nan
#             to_closest_6_time_interval_median = np.nan
#         if (7 in actionType_arr):
#             index_7 = np.where(actionType_arr == 7)[0]
#             closest_7_time = actionTime_arr[index_7[-1]]
#             to_closest_7_time = actionTime_arr[-1] - actionTime_arr[index_7[-1]]
#
#             to_closest_7_dist = arr_length - 1 - index_7[-1]
#             to_closest_7_time_interval_list = []
#             if (index_7[-1] == arr_length - 1):
#                 to_closest_7_time_interval_mean = np.nan
#                 to_closest_7_time_interval_var = np.nan
#                 to_closest_7_time_interval_min = np.nan
#                 to_closest_7_time_interval_max = np.nan
#                 to_closest_7_time_interval_median = np.nan
#
#             else:
#                 for i in range(index_7[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_7_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_7_time_interval_arr = np.asarray(to_closest_7_time_interval_list)
#                 to_closest_7_time_interval_mean = round(np.mean(to_closest_7_time_interval_arr), 2) if (
#                         len(to_closest_7_time_interval_arr) > 1) else np.nan
#                 to_closest_7_time_interval_min = np.min(to_closest_7_time_interval_arr)
#                 to_closest_7_time_interval_var = round(np.var(to_closest_7_time_interval_arr), 2) if (
#                         len(to_closest_7_time_interval_arr) > 1) else np.nan
#                 to_closest_7_time_interval_max = np.max(to_closest_7_time_interval_arr)
#                 to_closest_7_time_interval_median = np.median(to_closest_7_time_interval_arr)
#
#         else:
#             closest_7_time = np.nan
#             to_closest_7_time = np.nan
#             to_closest_7_dist = np.nan
#             to_closest_7_time_interval_mean = np.nan
#             to_closest_7_time_interval_var = np.nan
#             to_closest_7_time_interval_min = np.nan
#             to_closest_7_time_interval_max = np.nan
#             to_closest_7_time_interval_median = np.nan
#         if (8 in actionType_arr):
#             index_8 = np.where(actionType_arr == 8)[0]
#             closest_8_time = actionTime_arr[index_8[-1]]
#             to_closest_8_time = actionTime_arr[-1] - actionTime_arr[index_8[-1]]
#             to_closest_8_dist = arr_length - 1 - index_8[-1]
#             to_closest_8_time_interval_list = []
#             if (index_8[-1] == arr_length - 1):
#                 to_closest_8_time_interval_mean = np.nan
#                 to_closest_8_time_interval_var = np.nan
#                 to_closest_8_time_interval_min = np.nan
#                 to_closest_8_time_interval_max = np.nan
#                 to_closest_8_time_interval_median = np.nan
#
#             else:
#                 for i in range(index_8[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_8_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_8_time_interval_arr = np.asarray(to_closest_8_time_interval_list)
#                 to_closest_8_time_interval_mean = round(np.mean(to_closest_8_time_interval_arr), 2) if (
#                         len(to_closest_8_time_interval_arr) > 1) else np.nan
#                 to_closest_8_time_interval_min = np.min(to_closest_8_time_interval_arr)
#                 to_closest_8_time_interval_var = round(np.var(to_closest_8_time_interval_arr), 2) if (
#                         len(to_closest_8_time_interval_arr) > 1) else np.nan
#                 to_closest_8_time_interval_max = np.max(to_closest_8_time_interval_arr)
#                 to_closest_8_time_interval_median = np.median(to_closest_8_time_interval_arr)
#
#         else:
#             closest_8_time = np.nan
#             to_closest_8_time = np.nan
#             to_closest_8_dist = np.nan
#             to_closest_8_time_interval_mean = np.nan
#             to_closest_8_time_interval_var = np.nan
#             to_closest_8_time_interval_min = np.nan
#             to_closest_8_time_interval_max = np.nan
#             to_closest_8_time_interval_median = np.nan
#         if (9 in actionType_arr):
#             index_9 = np.where(actionType_arr == 9)[0]
#             closest_9_time = actionTime_arr[index_9[-1]]
#             to_closest_9_time = actionTime_arr[-1] - actionTime_arr[index_9[-1]]
#             to_closest_9_dist = arr_length - 1 - index_9[-1]
#             to_closest_9_time_interval_list = []
#             if (index_9[-1] == arr_length - 1):
#                 to_closest_9_time_interval_mean = np.nan
#                 to_closest_9_time_interval_var = np.nan
#                 to_closest_9_time_interval_min = np.nan
#                 to_closest_9_time_interval_max = np.nan
#                 to_closest_9_time_interval_median = np.nan
#                 to_closest_9_time_interval_mean_multi_var = np.nan
#             else:
#                 for i in range(index_9[-1], arr_length):
#                     if (i + 1 == arr_length):
#                         break
#                     to_closest_9_time_interval_list.append(actionTime_arr[i + 1] - actionTime_arr[i])
#                 to_closest_9_time_interval_arr = np.asarray(to_closest_9_time_interval_list)
#                 to_closest_9_time_interval_mean = round(np.mean(to_closest_9_time_interval_arr), 2) if (
#                         len(to_closest_9_time_interval_arr) > 1) else np.nan
#                 to_closest_9_time_interval_var = round(np.var(to_closest_9_time_interval_arr), 2) if (
#                         len(to_closest_9_time_interval_arr) > 1) else np.nan
#                 to_closest_9_time_interval_min = np.min(to_closest_9_time_interval_arr)
#                 to_closest_9_time_interval_max = np.max(to_closest_9_time_interval_arr)
#                 to_closest_9_time_interval_median = np.median(to_closest_9_time_interval_arr)
#                 to_closest_9_time_interval_mean_multi_var = to_closest_9_time_interval_mean * to_closest_9_time_interval_mean
#         else:
#             closest_9_time = np.nan
#             to_closest_9_time = np.nan
#             to_closest_9_dist = np.nan
#             to_closest_9_time_interval_mean = np.nan
#             to_closest_9_time_interval_var = np.nan
#             to_closest_9_time_interval_min = np.nan
#             to_closest_9_time_interval_max = np.nan
#             to_closest_9_time_interval_median = np.nan
#             to_closest_9_time_interval_mean_multi_var = np.nan
#         # Todo 此处修改open文件名，to_closest_X_infomation_xxx.csv
#         to_closest_X_infomation = open(feature_file_path + 'to_closest_X_infomation_train.csv', 'a')
#         to_closest_X_infomation.write('\n')
#         to_closest_X_infomation.write(str(df['userid'].values[0]) + ',' +
#                                       str(closest_1_time) + ',' + str(to_closest_1_time) + ',' + str(
#             to_closest_1_dist) + ',' + str(
#             to_closest_1_time_interval_mean) + ',' + str(to_closest_1_time_interval_var) + ',' + str(
#             to_closest_1_time_interval_min) + ',' + str(to_closest_1_time_interval_max) + ',' + str(
#             to_closest_1_time_interval_median) + ',' +
#                                       str(closest_2_time) + ',' + str(to_closest_2_time) + ',' + str(
#             to_closest_2_dist) + ',' + str(
#             to_closest_2_time_interval_mean) + ',' + str(
#             to_closest_2_time_interval_var) + ',' + str(to_closest_2_time_interval_min) + ',' + str(
#             to_closest_2_time_interval_max) + ',' + str(to_closest_2_time_interval_median) + ',' +
#                                       str(closest_3_time) + ',' + str(to_closest_3_time) + ',' + str(
#             to_closest_3_dist) + ',' + str(
#             to_closest_3_time_interval_mean) + ',' + str(
#             to_closest_3_time_interval_var) + ',' + str(to_closest_3_time_interval_min) + ',' + str(
#             to_closest_3_time_interval_max) + ',' + str(to_closest_3_time_interval_median) + ',' +
#                                       str(closest_4_time) + ',' + str(to_closest_4_time) + ',' + str(
#             to_closest_4_dist) + ',' + str(
#             to_closest_4_time_interval_mean) + ',' + str(
#             to_closest_4_time_interval_var) + ',' + str(to_closest_4_time_interval_min) + ',' + str(
#             to_closest_4_time_interval_max) + ',' + str(to_closest_4_time_interval_median) + ',' +
#                                       str(closest_5_time) + ',' + str(to_closest_5_time) + ',' + str(
#             to_closest_5_dist) + ',' + str(
#             to_closest_5_time_interval_mean) + ',' + str(
#             to_closest_5_time_interval_var) + ',' + str(to_closest_5_time_interval_min) + ',' + str(
#             to_closest_5_time_interval_max) + ',' + str(to_closest_5_time_interval_median) + ',' +
#                                       str(closest_6_time) + ',' + str(to_closest_6_time) + ',' + str(
#             to_closest_6_dist) + ',' + str(
#             to_closest_6_time_interval_mean) + ',' + str(
#             to_closest_6_time_interval_var) + ',' + str(to_closest_6_time_interval_min) + ',' + str(
#             to_closest_6_time_interval_max) + ',' + str(to_closest_6_time_interval_median) + ',' +
#                                       str(closest_7_time) + ',' + str(to_closest_7_time) + ',' + str(
#             to_closest_7_dist) + ',' + str(
#             to_closest_7_time_interval_mean) + ',' + str(
#             to_closest_7_time_interval_var) + ',' + str(to_closest_7_time_interval_min) + ',' + str(
#             to_closest_7_time_interval_max) + ',' + str(to_closest_7_time_interval_median) + ',' +
#                                       str(closest_8_time) + ',' + str(to_closest_8_time) + ',' + str(
#             to_closest_8_dist) + ',' + str(
#             to_closest_8_time_interval_mean) + ',' + str(
#             to_closest_8_time_interval_var) + ',' + str(to_closest_8_time_interval_min) + ',' + str(
#             to_closest_8_time_interval_max) + ',' + str(to_closest_8_time_interval_median) + ',' +
#                                       str(closest_9_time) + ',' + str(to_closest_9_time) + ',' + str(
#             to_closest_9_dist) + ',' + str(
#             to_closest_9_time_interval_mean) + ',' + str(
#             to_closest_9_time_interval_var) + ',' + str(to_closest_9_time_interval_min) + ',' + str(
#             to_closest_9_time_interval_max) + ',' + str(to_closest_9_time_interval_median) + ',' + str(
#             to_closest_9_time_interval_mean_multi_var))
#
#         to_closest_X_infomation.close()
#     num_to_closest_X_DistandTime += 1
#     return df
#
#
# num_to_closest_X_DistandTime = 0
# num_count = 0
# to_closest_X_infomation_train = open(feature_file_path + 'to_closest_X_infomation_train.csv', 'w')
# to_closest_X_infomation_train.write(
#     'userid' + ','
#                'closest_1_time' + ',' + 'to_closest_1_time' + ',' + 'to_closest_1_dist' + ',' + 'to_closest_1_time_interval_mean' + ',' + 'to_closest_1_time_interval_var' + ',' + 'to_closest_1_time_interval_min' + ',' + 'to_closest_1_time_interval_max' + ',' + 'to_closest_1_time_interval_median' + ',' +
#     'closest_2_time' + ',' + 'to_closest_2_time' + ',' + 'to_closest_2_dist' + ',' + 'to_closest_2_time_interval_mean' + ',' + 'to_closest_2_time_interval_var' + ',' + 'to_closest_2_time_interval_min' + ',' + 'to_closest_2_time_interval_max' + ',' + 'to_closest_2_time_interval_median' + ',' +
#     'closest_3_time' + ',' + 'to_closest_3_time' + ',' + 'to_closest_3_dist' + ',' + 'to_closest_3_time_interval_mean' + ',' + 'to_closest_3_time_interval_var' + ',' + 'to_closest_3_time_interval_min' + ',' + 'to_closest_3_time_interval_max' + ',' + 'to_closest_3_time_interval_median' + ',' +
#     'closest_4_time' + ',' + 'to_closest_4_time' + ',' + 'to_closest_4_dist' + ',' + 'to_closest_4_time_interval_mean' + ',' + 'to_closest_4_time_interval_var' + ',' + 'to_closest_4_time_interval_min' + ',' + 'to_closest_4_time_interval_max' + ',' + 'to_closest_4_time_interval_median' + ',' +
#     'closest_5_time' + ',' + 'to_closest_5_time' + ',' + 'to_closest_5_dist' + ',' + 'to_closest_5_time_interval_mean' + ',' + 'to_closest_5_time_interval_var' + ',' + 'to_closest_5_time_interval_min' + ',' + 'to_closest_5_time_interval_max' + ',' + 'to_closest_5_time_interval_median' + ',' +
#     'closest_6_time' + ',' + 'to_closest_6_time' + ',' + 'to_closest_6_dist' + ',' + 'to_closest_6_time_interval_mean' + ',' + 'to_closest_6_time_interval_var' + ',' + 'to_closest_6_time_interval_min' + ',' + 'to_closest_6_time_interval_max' + ',' + 'to_closest_6_time_interval_median' + ',' +
#     'closest_7_time' + ',' + 'to_closest_7_time' + ',' + 'to_closest_7_dist' + ',' + 'to_closest_7_time_interval_mean' + ',' + 'to_closest_7_time_interval_var' + ',' + 'to_closest_7_time_interval_min' + ',' + 'to_closest_7_time_interval_max' + ',' + 'to_closest_7_time_interval_median' + ',' +
#     'closest_8_time' + ',' + 'to_closest_8_time' + ',' + 'to_closest_8_dist' + ',' + 'to_closest_8_time_interval_mean' + ',' + 'to_closest_8_time_interval_var' + ',' + 'to_closest_8_time_interval_min' + ',' + 'to_closest_8_time_interval_max' + ',' + 'to_closest_8_time_interval_median' + ',' +
#     'closest_9_time' + ',' + 'to_closest_9_time' + ',' + 'to_closest_9_dist' + ',' + 'to_closest_9_time_interval_mean' + ',' + 'to_closest_9_time_interval_var' + ',' + 'to_closest_9_time_interval_min' + ',' + 'to_closest_9_time_interval_max' + ',' + 'to_closest_9_time_interval_median' + ',' +
#     'to_closest_9_time_interval_mean_multi_var'
# )
# to_closest_X_infomation_train.close()
# print('有' + str(len(action_train['userid'].unique())))
# # action_train = action_train[action_train['userid'] == 100000000013]
# action_train.groupby('userid').apply(fun_to_closest_X_DistandTime)

# num_to_closest_X_DistandTime = 0
# num_count = 0
# to_closest_X_infomation_test = open(feature_file_path + 'to_closest_X_infomation_test.csv', 'w')
# to_closest_X_infomation_test.write(
#     'userid' + ','
#                'closest_1_time' + ',' + 'to_closest_1_time' + ',' + 'to_closest_1_dist' + ',' + 'to_closest_1_time_interval_mean' + ',' + 'to_closest_1_time_interval_var' + ',' + 'to_closest_1_time_interval_min' + ',' + 'to_closest_1_time_interval_max' + ',' + 'to_closest_1_time_interval_median' + ',' +
#     'closest_2_time' + ',' + 'to_closest_2_time' + ',' + 'to_closest_2_dist' + ',' + 'to_closest_2_time_interval_mean' + ',' + 'to_closest_2_time_interval_var' + ',' + 'to_closest_2_time_interval_min' + ',' + 'to_closest_2_time_interval_max' + ',' + 'to_closest_2_time_interval_median' + ',' +
#     'closest_3_time' + ',' + 'to_closest_3_time' + ',' + 'to_closest_3_dist' + ',' + 'to_closest_3_time_interval_mean' + ',' + 'to_closest_3_time_interval_var' + ',' + 'to_closest_3_time_interval_min' + ',' + 'to_closest_3_time_interval_max' + ',' + 'to_closest_3_time_interval_median' + ',' +
#     'closest_4_time' + ',' + 'to_closest_4_time' + ',' + 'to_closest_4_dist' + ',' + 'to_closest_4_time_interval_mean' + ',' + 'to_closest_4_time_interval_var' + ',' + 'to_closest_4_time_interval_min' + ',' + 'to_closest_4_time_interval_max' + ',' + 'to_closest_4_time_interval_median' + ',' +
#     'closest_5_time' + ',' + 'to_closest_5_time' + ',' + 'to_closest_5_dist' + ',' + 'to_closest_5_time_interval_mean' + ',' + 'to_closest_5_time_interval_var' + ',' + 'to_closest_5_time_interval_min' + ',' + 'to_closest_5_time_interval_max' + ',' + 'to_closest_5_time_interval_median' + ',' +
#     'closest_6_time' + ',' + 'to_closest_6_time' + ',' + 'to_closest_6_dist' + ',' + 'to_closest_6_time_interval_mean' + ',' + 'to_closest_6_time_interval_var' + ',' + 'to_closest_6_time_interval_min' + ',' + 'to_closest_6_time_interval_max' + ',' + 'to_closest_6_time_interval_median' + ',' +
#     'closest_7_time' + ',' + 'to_closest_7_time' + ',' + 'to_closest_7_dist' + ',' + 'to_closest_7_time_interval_mean' + ',' + 'to_closest_7_time_interval_var' + ',' + 'to_closest_7_time_interval_min' + ',' + 'to_closest_7_time_interval_max' + ',' + 'to_closest_7_time_interval_median' + ',' +
#     'closest_8_time' + ',' + 'to_closest_8_time' + ',' + 'to_closest_8_dist' + ',' + 'to_closest_8_time_interval_mean' + ',' + 'to_closest_8_time_interval_var' + ',' + 'to_closest_8_time_interval_min' + ',' + 'to_closest_8_time_interval_max' + ',' + 'to_closest_8_time_interval_median' + ',' +
#     'closest_9_time' + ',' + 'to_closest_9_time' + ',' + 'to_closest_9_dist' + ',' + 'to_closest_9_time_interval_mean' + ',' + 'to_closest_9_time_interval_var' + ',' + 'to_closest_9_time_interval_min' + ',' + 'to_closest_9_time_interval_max' + ',' + 'to_closest_9_time_interval_median' + ',' +
#     'to_closest_9_time_interval_mean_multi_var'
# )
# to_closest_X_infomation_test.close()
# print('有' + str(len(action_test['userid'].unique())))
# action_test.groupby('userid').apply(fun_to_closest_X_DistandTime)

# # 用户点击X与总点击量比值
#
# action_train = pd.read_csv(traindata_path + "action_train.csv")
# action_test = pd.read_csv(testdata_path + "action_test.csv")
#
#
# def fun_user_X_click_rate_train(df):
#     global num_click_rate
#     global userid_click_X_rate_train
#     global click1_num_train
#     global click2_num_train
#     global click3_num_train
#     global click4_num_train
#     global click5_num_train
#     global click6_num_train
#     global click7_num_train
#     global click8_num_train
#     global click9_num_train
#
#     if (num_click_rate != 0):
#         userid = df['userid'].values[0]
#         user_clickX_num = df.shape[0]
#         print('user_clickX_num: ' + str(user_clickX_num))
#         if (1 in df['actionType'].values):
#             print('click_1_num: ' + str(df[df['actionType'] == 1].shape[0]))
#             click1_rate = df[df['actionType'] == 1].shape[0] / user_clickX_num
#         else:
#             click1_rate = np.nan
#         if (2 in df['actionType'].values):
#             print('click_2_num: ' + str(df[df['actionType'] == 2].shape[0]))
#             click2_rate = df[df['actionType'] == 2].shape[0] / user_clickX_num
#         else:
#             click2_rate = np.nan
#         if (3 in df['actionType'].values):
#             print('click_3_num: ' + str(df[df['actionType'] == 3].shape[0]))
#
#             click3_rate = df[df['actionType'] == 3].shape[0] / user_clickX_num
#         else:
#             click3_rate = np.nan
#         if (4 in df['actionType'].values):
#             print('click_4_num: ' + str(df[df['actionType'] == 4].shape[0]))
#
#             click4_rate = df[df['actionType'] == 4].shape[0] / user_clickX_num
#         else:
#             click4_rate = np.nan
#         if (5 in df['actionType'].values):
#             print('click_5_num: ' + str(df[df['actionType'] == 5].shape[0]))
#
#             click5_rate = df[df['actionType'] == 5].shape[0] / user_clickX_num
#         else:
#             click5_rate = np.nan
#         if (6 in df['actionType'].values):
#             print('click_6_num: ' + str(df[df['actionType'] == 6].shape[0]))
#
#             click6_rate = df[df['actionType'] == 6].shape[0] / user_clickX_num
#         else:
#             click6_rate = np.nan
#         if (7 in df['actionType'].values):
#             print('click_7_num: ' + str(df[df['actionType'] == 7].shape[0]))
#
#             click7_rate = df[df['actionType'] == 7].shape[0] / user_clickX_num
#         else:
#             click7_rate = np.nan
#         if (8 in df['actionType'].values):
#             print('click_8_num: ' + str(df[df['actionType'] == 8].shape[0]))
#
#             click8_rate = df[df['actionType'] == 8].shape[0] / user_clickX_num
#         else:
#             click8_rate = np.nan
#         if (9 in df['actionType'].values):
#             print('click_9_num: ' + str(df[df['actionType'] == 9].shape[0]))
#
#             click9_rate = df[df['actionType'] == 9].shape[0] / user_clickX_num
#         else:
#             click9_rate = np.nan
#         userid_click_X_rate_train[userid]['click1_rate'] = click1_rate
#         userid_click_X_rate_train[userid]['click2_rate'] = click2_rate
#         userid_click_X_rate_train[userid]['click3_rate'] = click3_rate
#         userid_click_X_rate_train[userid]['click4_rate'] = click4_rate
#         userid_click_X_rate_train[userid]['click5_rate'] = click5_rate
#         userid_click_X_rate_train[userid]['click6_rate'] = click6_rate
#         userid_click_X_rate_train[userid]['click7_rate'] = click7_rate
#         userid_click_X_rate_train[userid]['click8_rate'] = click8_rate
#         userid_click_X_rate_train[userid]['click9_rate'] = click9_rate
#
#     num_click_rate += 1
#     return df
#
#
# click1_num_train = action_train[action_train['actionType'] == 1].shape[0]
# click2_num_train = action_train[action_train['actionType'] == 2].shape[0]
# click3_num_train = action_train[action_train['actionType'] == 3].shape[0]
# click4_num_train = action_train[action_train['actionType'] == 4].shape[0]
# click5_num_train = action_train[action_train['actionType'] == 5].shape[0]
# click6_num_train = action_train[action_train['actionType'] == 6].shape[0]
# click7_num_train = action_train[action_train['actionType'] == 7].shape[0]
# click8_num_train = action_train[action_train['actionType'] == 8].shape[0]
# click9_num_train = action_train[action_train['actionType'] == 9].shape[0]
# clickX_num = action_train.shape[0]
# num_click_rate = 0
# userid_click_X_rate_train = defaultdict(lambda: defaultdict(lambda: 0))
# print('constructing userid_click_X_rate_train ')
# # action_train = action_train[action_train['userid'] == 100000000013]
# action_train.groupby('userid').apply(fun_user_X_click_rate_train)
# clickX_rate_df_train = pd.DataFrame.from_dict(userid_click_X_rate_train, orient='index').reset_index()
# clickX_rate_df_train.rename(columns={'index': 'userid'}, inplace=True)
#
# # scaler = preprocessing.MinMaxScaler()
# # userid_df = clickX_rate_df_train.iloc[:, 0]
# # del clickX_rate_df_train['userid']
# # middle_arr = scaler.fit_transform(clickX_rate_df_train)
# # ori_columns = clickX_rate_df_train.columns.values
# # df = pd.DataFrame(middle_arr, columns=ori_columns)
# # clickX_rate_df_train = pd.concat([userid_df, df], axis=1)
#
# clickX_rate_df_train.to_csv(feature_file_path + 'clickX_rate_train.csv', index=False)
#
#
# def fun_user_X_click_rate_test(df):
#     global num_click_rate
#     global userid_click_X_rate_test
#     global click1_num_test
#     global click2_num_test
#     global click3_num_test
#     global click4_num_test
#     global click5_num_test
#     global click6_num_test
#     global click7_num_test
#     global click8_num_test
#     global click9_num_test
#
#     if (num_click_rate != 0):
#         userid = df['userid'].values[0]
#         user_clickX_num = df.shape[0]
#         print('user_clickX_num: ' + str(user_clickX_num))
#         if (1 in df['actionType'].values):
#             click1_rate = df[df['actionType'] == 1].shape[0] / user_clickX_num
#         else:
#             click1_rate = np.nan
#         if (2 in df['actionType'].values):
#             click2_rate = df[df['actionType'] == 2].shape[0] / user_clickX_num
#         else:
#             click2_rate = np.nan
#         if (3 in df['actionType'].values):
#             click3_rate = df[df['actionType'] == 3].shape[0] / user_clickX_num
#         else:
#             click3_rate = np.nan
#         if (4 in df['actionType'].values):
#             click4_rate = df[df['actionType'] == 4].shape[0] / user_clickX_num
#         else:
#             click4_rate = np.nan
#         if (5 in df['actionType'].values):
#             click5_rate = df[df['actionType'] == 5].shape[0] / user_clickX_num
#         else:
#             click5_rate = np.nan
#         if (6 in df['actionType'].values):
#             click6_rate = df[df['actionType'] == 6].shape[0] / user_clickX_num
#         else:
#             click6_rate = np.nan
#         if (7 in df['actionType'].values):
#             click7_rate = df[df['actionType'] == 7].shape[0] / user_clickX_num
#         else:
#             click7_rate = np.nan
#         if (8 in df['actionType'].values):
#             click8_rate = df[df['actionType'] == 8].shape[0] / user_clickX_num
#         else:
#             click8_rate = np.nan
#         if (9 in df['actionType'].values):
#             click9_rate = df[df['actionType'] == 9].shape[0] / user_clickX_num
#         else:
#             click9_rate = np.nan
#         userid_click_X_rate_test[userid]['click1_rate'] = click1_rate
#         userid_click_X_rate_test[userid]['click2_rate'] = click2_rate
#         userid_click_X_rate_test[userid]['click3_rate'] = click3_rate
#         userid_click_X_rate_test[userid]['click4_rate'] = click4_rate
#         userid_click_X_rate_test[userid]['click5_rate'] = click5_rate
#         userid_click_X_rate_test[userid]['click6_rate'] = click6_rate
#         userid_click_X_rate_test[userid]['click7_rate'] = click7_rate
#         userid_click_X_rate_test[userid]['click8_rate'] = click8_rate
#         userid_click_X_rate_test[userid]['click9_rate'] = click9_rate
#
#     num_click_rate += 1
#     return df
#
#
# click1_num_test = action_test[action_test['actionType'] == 1].shape[0]
# click2_num_test = action_test[action_test['actionType'] == 2].shape[0]
# click3_num_test = action_test[action_test['actionType'] == 3].shape[0]
# click4_num_test = action_test[action_test['actionType'] == 4].shape[0]
# click5_num_test = action_test[action_test['actionType'] == 5].shape[0]
# click6_num_test = action_test[action_test['actionType'] == 6].shape[0]
# click7_num_test = action_test[action_test['actionType'] == 7].shape[0]
# click8_num_test = action_test[action_test['actionType'] == 8].shape[0]
# click9_num_test = action_test[action_test['actionType'] == 9].shape[0]
# clickX_num = action_test.shape[0]
# userid_click_X_rate_test = defaultdict(lambda: defaultdict(lambda: 0))
# print('constructing userid_click_X_rate_test ')
# action_test.groupby('userid').apply(fun_user_X_click_rate_test)
# clickX_rate_df_test = pd.DataFrame.from_dict(userid_click_X_rate_test, orient='index').reset_index()
# clickX_rate_df_test.rename(columns={'index': 'userid'}, inplace=True)
#
# # scaler = preprocessing.MinMaxScaler()
# # userid_df = clickX_rate_df_test.iloc[:, 0]
# # del clickX_rate_df_test['userid']
# # middle_arr = scaler.fit_transform(clickX_rate_df_test)
# # ori_columns = clickX_rate_df_test.columns.values
# # df = pd.DataFrame(middle_arr, columns=ori_columns)
# # clickX_rate_df_test = pd.concat([userid_df, df], axis=1)
#
# clickX_rate_df_test.to_csv(feature_file_path + 'clickX_rate_test.csv', index=False)
