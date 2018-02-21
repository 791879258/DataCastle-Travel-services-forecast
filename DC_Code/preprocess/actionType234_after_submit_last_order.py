#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-03 10:16:20
# @Author  : guanglinzhou (xdzgl812@163.com)


# 统计用户在提交最后一次订单后继续浏览商品的次数
# 用户包括(history0_future0,即history_orderType0——future_orderType0,history_0_future1即history_orderType0——future_orderType1)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

################
project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'
traindata_path = project_path + "data/train/"
traindata_output_path = project_path + "preprocess/"
middle_file_path = project_path + 'preprocess/middle_file/'
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderFuture_train = pd.read_csv(traindata_path + "orderFuture_train.csv")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
userComment_train = pd.read_csv(traindata_path + "userComment_train.csv")
userProfile_train = pd.read_csv(traindata_path + "userProfile_train.csv")
history_action_merge_by_time_train = pd.read_csv(
    project_path + 'preprocess/middle_file/history_action_merge_by_time_train.csv')

# orderHistory_train中orderType全为0的用户:orderHistory_orderType_all_zero
orderHistory_orderType_all_zero = pd.DataFrame(
    columns=[['userid', 'orderid', 'orderTime', 'orderType', 'city', 'country', 'continent']])
orderHistory_orderType_all_zero_num = 0


def fun_construct_orderHistory_orderType_all_zero(df):
    global orderHistory_orderType_all_zero
    global orderHistory_orderType_all_zero_num
    if (1 not in df['orderType'].values):
        if (orderHistory_orderType_all_zero_num != 0):  # 由于groupby()会传递第一组数据两次，所以跳过第一次
            orderHistory_orderType_all_zero = pd.concat([orderHistory_orderType_all_zero, df], axis=0)
    orderHistory_orderType_all_zero_num += 1


orderHistory_train.groupby('userid').apply(fun_construct_orderHistory_orderType_all_zero)

# orderFuture_train中orderType为1的用户：orderfuture_orderType_one
orderfuture_orderType_one = orderFuture_train[orderFuture_train['orderType'] == 1]
userid_list_of_history0_future1 = list(
    set(orderHistory_orderType_all_zero['userid']) & set(orderfuture_orderType_one['userid']))
userid_list_of_history0_future1_train = open(middle_file_path + 'userid_list_of_history0_future1_train.txt', 'w')
for item in userid_list_of_history0_future1:
    userid_list_of_history0_future1_train.write("%s\n" % item)

userid_list_of_history0_future0 = list(
    set(orderHistory_orderType_all_zero['userid']) - set(userid_list_of_history0_future1))
userid_list_of_history0_future0_train = open(middle_file_path + 'userid_list_of_history0_future0_train.txt', 'w')
for item in userid_list_of_history0_future0:
    userid_list_of_history0_future0_train.write("%s\n" % item)
# dict{rating:userid}
rating_userid_dict_history0_future1 = defaultdict(lambda: [])
for userid in userid_list_of_history0_future1:
    if (userid in userComment_train['userid'].values):
        Comment_rating = userComment_train[userComment_train['userid'] == userid]['rating']
        rating_userid_dict_history0_future1[Comment_rating.values[0]].append(userid)
    else:
        rating_userid_dict_history0_future1['NaN'].append(userid)
print('rating_userid_dict_history0_future1.keys():\n')
print(rating_userid_dict_history0_future1.keys())

for rating in rating_userid_dict_history0_future1.keys():
    print('rating ' + str(rating) + ' has ' + str(len(rating_userid_dict_history0_future1[rating])) + ' userid')
rating_userid_dict_history0_future0 = defaultdict(lambda: [])
for userid in userid_list_of_history0_future0:
    if (userid in userComment_train['userid'].values):
        Comment_rating = userComment_train[userComment_train['userid'] == userid]['rating']
        rating_userid_dict_history0_future0[Comment_rating.values[0]].append(userid)
    else:
        rating_userid_dict_history0_future0['NaN'].append(userid)
print('rating_userid_dict_history0_future0.keys():\n')
print(rating_userid_dict_history0_future0.keys())

for rating in rating_userid_dict_history0_future0.keys():
    print('rating ' + str(rating) + ' has ' + str(len(rating_userid_dict_history0_future0[rating])) + ' userid')

userid_list_history0_future1_rating_larger3 = []
userid_list_history0_future1_rating_larger3.extend(rating_userid_dict_history0_future1[5])
userid_list_history0_future1_rating_larger3.extend(rating_userid_dict_history0_future1[4])
print('len(userid_list_history0_future1_rating_larger3):')
print(len(userid_list_history0_future1_rating_larger3))

userid_actionType_num_dict_history0_future1 = defaultdict(lambda: defaultdict(lambda: 0))
for userid in userid_list_history0_future1_rating_larger3:
    last_orderTime_index = history_action_merge_by_time_train[
        (history_action_merge_by_time_train['userid'] == userid) & (
                history_action_merge_by_time_train['actionTime_zero_orderTime_one'] == 1)].index.values.max()
    last_userid_action_index = history_action_merge_by_time_train[
        history_action_merge_by_time_train['userid'] == userid].index.values.max()

    last_range_df = history_action_merge_by_time_train.iloc[last_orderTime_index + 1:last_userid_action_index + 1, :]
    #     print(last_range_df)
    for actionType in last_range_df['actionType'].values:
        userid_actionType_num_dict_history0_future1[userid][actionType] += 1
Type234_num = 0
for userid in userid_actionType_num_dict_history0_future1.keys():
    Type234_num += userid_actionType_num_dict_history0_future1[userid][2] + \
                   userid_actionType_num_dict_history0_future1[userid][3] + \
                   userid_actionType_num_dict_history0_future1[userid][4]
print("在history0_future1,用户提交最后一次订单后，继续浏览商品的数量为：" + str(Type234_num) + '\n')

userid_list_history0_future0_rating_larger3 = []
userid_list_history0_future0_rating_larger3.extend(rating_userid_dict_history0_future0[5])
userid_list_history0_future0_rating_larger3.extend(rating_userid_dict_history0_future0[4])
print('len(userid_list_history0_future0_rating_larger3):')
print(len(userid_list_history0_future0_rating_larger3))

userid_actionType_num_dict_history0_future0 = defaultdict(lambda: defaultdict(lambda: 0))
for userid in userid_list_history0_future0_rating_larger3:
    last_orderTime_index = history_action_merge_by_time_train[
        (history_action_merge_by_time_train['userid'] == userid) & (
                history_action_merge_by_time_train['actionTime_zero_orderTime_one'] == 1)].index.values.max()
    last_userid_action_index = history_action_merge_by_time_train[
        history_action_merge_by_time_train['userid'] == userid].index.values.max()

    last_range_df = history_action_merge_by_time_train.iloc[last_orderTime_index + 1:last_userid_action_index + 1, :]
    #     print(last_range_df)
    for actionType in last_range_df['actionType'].values:
        userid_actionType_num_dict_history0_future0[userid][actionType] += 1
Type234_num = 0
for userid in userid_actionType_num_dict_history0_future0.keys():
    Type234_num += userid_actionType_num_dict_history0_future0[userid][2] + \
                   userid_actionType_num_dict_history0_future0[userid][3] + \
                   userid_actionType_num_dict_history0_future0[userid][4]
print("在history0_future0,用户提交最后一次订单后，继续浏览商品的数量为：" + str(Type234_num) + '\n')
