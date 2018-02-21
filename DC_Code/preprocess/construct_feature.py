#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-29 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import datetime
import gc

# 特征均可通过merge操作，on='userid'来合并到表xxxdata_output.csv中

###########-----------------------读表----------------------------###########
project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_path = project_path + "data/train/"
traindata_path = project_path + "data/train/"
testdata_path = project_path + "data/test/"
feature_file_path = project_path + 'preprocess/feature_file/'


###########-----------------------function----------------------------###########
def divide(x, y):
    return float(x) / y


def min_max_normalize(df, name):
    # 归一化
    max_number = df[name].max()
    min_number = df[name].min()
    # assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
    df[name] = df[name].map(lambda x: round(float(x - min_number) / float(max_number - min_number), 2))
    return df


###########-----------------------构建特征----------------------------###########
# 1、特征ever_purchase_orderType1表示userid是否购买过精品旅游服务，1表示有，0表示无。
# 0	1.0	100000001023
# 1	1.0	100000001505
# 2	1.0	100000003461
print("constructing feature ever_purchase_orderType1")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
orderHistory_train.drop(orderHistory_train.columns[[1, 2, 4, 5, 6]], axis=1, inplace=True)
ever_purchase_orderType1 = orderHistory_train[orderHistory_train['orderType'] == 1]['userid'].unique()
ever_purchase_orderType1 = pd.DataFrame(
    {'userid': ever_purchase_orderType1, 'ever_purchase_orderType1': np.ones(len(ever_purchase_orderType1))})
ever_purchase_orderType1.fillna(0, inplace=True)
ever_purchase_orderType1.to_csv(feature_file_path + 'ever_purchase_orderType1_train.csv', index=False)

orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")
orderHistory_test.drop(orderHistory_test.columns[[1, 2, 4, 5, 6]], axis=1, inplace=True)
ever_purchase_orderType1 = orderHistory_test[orderHistory_test['orderType'] == 1]['userid'].unique()
ever_purchase_orderType1 = pd.DataFrame(
    {'userid': ever_purchase_orderType1, 'ever_purchase_orderType1': np.ones(len(ever_purchase_orderType1))})
ever_purchase_orderType1.fillna(0, inplace=True)
ever_purchase_orderType1.to_csv(feature_file_path + 'ever_purchase_orderType1_test.csv', index=False)

# 2、特征user_purchase_orderType01_times，表示用户是否多次购买过旅游服务(包括普通服务和精品旅游服务)
print("constructing feature user_purchase_orderType01_times")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
cnt_train = Counter()
for userid in np.array(orderHistory_train['userid']):
    cnt_train[userid] += 1
user_purchase_orderType01_times_train = pd.DataFrame(list(dict(cnt_train).items()),
                                                     columns=['userid', 'user_purchase_orderType01_times'])
user_purchase_orderType01_times_train.fillna(0, inplace=True)
user_purchase_orderType01_times_train.to_csv(feature_file_path + 'user_purchase_orderType01_times_train.csv',
                                             index=False)

orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")
cnt_test = Counter()
for userid in np.array(orderHistory_test['userid']):
    cnt_test[userid] += 1
user_purchase_orderType01_times_test = pd.DataFrame(list(dict(cnt_test).items()),
                                                    columns=['userid', 'user_purchase_orderType01_times'])
user_purchase_orderType01_times_test.fillna(0, inplace=True)
user_purchase_orderType01_times_test.to_csv(feature_file_path + 'user_purchase_orderType01_times_test.csv', index=False)

# 3、特征if_exist_rate_largerthan_2，表示用户评分记录中是否存在>=3的记录
#  user_orderType0_rate_largerthan_2，表示用户购买了普通服务且评分>=3的记录
userComment_train = pd.read_csv(traindata_path + "userComment_train.csv")
if_exist_rate_largerthan_2_train = userComment_train[userComment_train['rating'] >= 3]['userid']
if_exist_rate_largerthan_2_train = pd.DataFrame({'userid': if_exist_rate_largerthan_2_train,
                                                 'if_exist_rate_largerthan_2': np.ones(
                                                     len(if_exist_rate_largerthan_2_train))})
if_exist_rate_largerthan_2_train.fillna(0, inplace=True)
if_exist_rate_largerthan_2_train.to_csv(feature_file_path + 'if_exist_rate_largerthan_2_train.csv', index=False)
userComment_test = pd.read_csv(testdata_path + "userComment_test.csv")
if_exist_rate_largerthan_2_test = userComment_train[userComment_train['rating'] >= 3]['userid']
if_exist_rate_largerthan_2_test = pd.DataFrame({'userid': if_exist_rate_largerthan_2_test,
                                                'if_exist_rate_largerthan_2': np.ones(
                                                    len(if_exist_rate_largerthan_2_test))})
if_exist_rate_largerthan_2_test.fillna(0, inplace=True)
if_exist_rate_largerthan_2_test.to_csv(feature_file_path + 'if_exist_rate_largerthan_2_test.csv', index=False)

print("constructing feature user_orderType0_rate_largerthan_2")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")
userComment_train = pd.read_csv(traindata_path + "userComment_train.csv")
user_orderType0_rate_largerthan_2_train = pd.merge(orderHistory_train, userComment_train, on='userid', how='left',
                                                   suffixes=['history_orderid', 'comment_orderid'])
user_orderType0_rate_largerthan_2_train = user_orderType0_rate_largerthan_2_train[
    (user_orderType0_rate_largerthan_2_train['orderType'] == 0) & (
            user_orderType0_rate_largerthan_2_train['rating'] > 2)]
user_orderType0_rate_largerthan_2_train['user_orderType0_rate_largerthan_2'] = 1
user_orderType0_rate_largerthan_2_train = user_orderType0_rate_largerthan_2_train[
    ['userid', 'user_orderType0_rate_largerthan_2']]
user_orderType0_rate_largerthan_2_train.drop_duplicates(inplace=True)
user_orderType0_rate_largerthan_2_train.fillna(0, inplace=True)
user_orderType0_rate_largerthan_2_train.to_csv(feature_file_path + 'user_orderType0_rate_largerthan_2_train.csv',
                                               index=False)

orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")
userComment_test = pd.read_csv(testdata_path + "userComment_test.csv")
user_orderType0_rate_largerthan_2_test = pd.merge(orderHistory_test, userComment_test, on='userid', how='left',
                                                  suffixes=['history_orderid', 'comment_orderid'])
user_orderType0_rate_largerthan_2_test = user_orderType0_rate_largerthan_2_test[
    (user_orderType0_rate_largerthan_2_test['orderType'] == 0) & (user_orderType0_rate_largerthan_2_test['rating'] > 2)]
user_orderType0_rate_largerthan_2_test['user_orderType0_rate_largerthan_2'] = 1
user_orderType0_rate_largerthan_2_test = user_orderType0_rate_largerthan_2_test[
    ['userid', 'user_orderType0_rate_largerthan_2']]
user_orderType0_rate_largerthan_2_test.drop_duplicates(inplace=True)
user_orderType0_rate_largerthan_2_test.fillna(0, inplace=True)
user_orderType0_rate_largerthan_2_test.to_csv(feature_file_path + 'user_orderType0_rate_largerthan_2_test.csv',
                                              index=False)

# 5、 特征user_click_xxx
# user_click_num表示用户点击次数(行为数据中action_train.csv中userid出现次数)
# user_click_rate用户点击率(user点击占总点击数比率)
# user_purchase_num用户在购买次数history中出现的次数
# user_purchase_rate用户购买率(用户在购买次数history中出现的次数/点击次数)
# user_open_app_num用户打开app的次数
print('constructing feature  user_click_xxx')
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderHistory_train = pd.read_csv(traindata_path + 'orderHistory_train.csv')

cnt_user_click_num_train = Counter()
for userid in action_train['userid']:
    cnt_user_click_num_train[userid] += 1
user_click_train = pd.DataFrame.from_dict(dict(cnt_user_click_num_train), orient='index')
user_click_train.columns = ['user_click_num']
user_click_train['userid'] = user_click_train.index
user_click_train.reset_index(drop=True, inplace=True)
click_sum_train = sum(user_click_train['user_click_num'])
user_click_train['user_click_rate'] = user_click_train['user_click_num'].apply(
    lambda x: divide(x, click_sum_train))
user_click_train = min_max_normalize(user_click_train, 'user_click_rate')

user_purchase_num = orderHistory_train.groupby('userid').count()
user_purchase_num.drop(user_purchase_num.columns[[1, 2, 3, 4, 5]], axis=1, inplace=True)
user_purchase_num.reset_index(inplace=True)
user_purchase_num.rename(columns={'orderid': 'user_purchase_num'}, inplace=True)
user_click_train = pd.merge(user_click_train, user_purchase_num, how='left', on='userid').fillna(0)

action_train_actionType_equal1 = action_train[action_train['actionType'] == 1]
cnt_user_open_app_num = Counter()
for userid in action_train_actionType_equal1['userid']:
    cnt_user_open_app_num[userid] += 1
user_open_app_num = pd.DataFrame.from_dict(dict(cnt_user_open_app_num), orient='index')
user_open_app_num.columns = ['user_open_app_num']
user_open_app_num['userid'] = user_open_app_num.index
user_open_app_num.reset_index(drop=True, inplace=True)
user_click_train = pd.merge(user_click_train, user_open_app_num, how='left', on='userid').fillna(0)

user_click_train['user_purchase_rate'] = user_click_train.apply(
    lambda x: x['user_purchase_num'] / x['user_click_num'],
    axis=1)
# user_click_train = min_max_normalize(user_click_train, 'user_purchase_rate')
user_click_train.to_csv(feature_file_path + 'user_click_train.csv', index=False)

# test
action_test = pd.read_csv(testdata_path + "action_test.csv")
orderHistory_test = pd.read_csv(testdata_path + 'orderHistory_test.csv')

cnt_user_click_num_test = Counter()
for userid in action_test['userid']:
    cnt_user_click_num_test[userid] += 1
user_click_test = pd.DataFrame.from_dict(dict(cnt_user_click_num_test), orient='index')
user_click_test.columns = ['user_click_num']
user_click_test['userid'] = user_click_test.index
user_click_test.reset_index(drop=True, inplace=True)
click_sum_test = sum(user_click_test['user_click_num'])
user_click_test['user_click_rate'] = user_click_test['user_click_num'].apply(
    lambda x: divide(x, click_sum_test))
user_click_test = min_max_normalize(user_click_test, 'user_click_rate')

user_purchase_num = orderHistory_test.groupby('userid').count()
user_purchase_num.drop(user_purchase_num.columns[[1, 2, 3, 4, 5]], axis=1, inplace=True)
user_purchase_num.reset_index(inplace=True)
user_purchase_num.rename(columns={'orderid': 'user_purchase_num'}, inplace=True)
user_click_test = pd.merge(user_click_test, user_purchase_num, how='left', on='userid').fillna(0)

action_test_actionType_equal1 = action_test[action_test['actionType'] == 1]
cnt_user_open_app_num = Counter()
for userid in action_test_actionType_equal1['userid']:
    cnt_user_open_app_num[userid] += 1
user_open_app_num = pd.DataFrame.from_dict(dict(cnt_user_open_app_num), orient='index')
user_open_app_num.columns = ['user_open_app_num']
user_open_app_num['userid'] = user_open_app_num.index
user_open_app_num.reset_index(drop=True, inplace=True)
user_click_test = pd.merge(user_click_test, user_open_app_num, how='left', on='userid').fillna(0)

user_click_test['user_purchase_rate'] = user_click_test.apply(
    lambda x: x['user_purchase_num'] / x['user_click_num'],
    axis=1)
# user_click_test = min_max_normalize(user_click_test, 'user_purchase_rate')
user_click_test.to_csv(feature_file_path + 'user_click_test.csv', index=False)

# 特征 actionType234_after_submit_last_order，表示用户在提交最后一次订单后继续浏览产品的次数。
print('constructing feature actionType234_after_submit_last_order')
history_action_merge_by_time_train = pd.read_csv(
    project_path + 'preprocess/middle_file/history_action_merge_by_time_train.csv')
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")

userid_actionType234_num_dict_trian = defaultdict(lambda: defaultdict(lambda: 0))
userid_set_submitted_order_train = set(orderHistory_train['userid'])
for userid in userid_set_submitted_order_train:
    last_orderTime_index = history_action_merge_by_time_train[
        (history_action_merge_by_time_train['userid'] == userid) & (
                history_action_merge_by_time_train['actionTime_zero_orderTime_one'] == 1)].index.values.max()
    last_actionTime_index = history_action_merge_by_time_train[
        history_action_merge_by_time_train['userid'] == userid].index.values.max()

    last_range_df = history_action_merge_by_time_train.iloc[last_orderTime_index + 1:last_actionTime_index + 1, :]

    for actionType in last_range_df['actionType'].values:
        userid_actionType234_num_dict_trian[userid][actionType] += 1

userid_num_of_actionType234_dict_train = defaultdict(lambda: 0)
for userid in userid_actionType234_num_dict_trian.keys():
    Type234_num = 0
    Type234_num += (userid_actionType234_num_dict_trian[userid][2] + \
                    userid_actionType234_num_dict_trian[userid][3] + \
                    userid_actionType234_num_dict_trian[userid][4])
    userid_num_of_actionType234_dict_train[userid] = Type234_num

# print(userid_num_of_actionType234_dict_train)
actionType234_after_submit_last_order_train = pd.DataFrame.from_dict(userid_num_of_actionType234_dict_train,
                                                                     orient='index')
actionType234_after_submit_last_order_train.reset_index(inplace=True)
actionType234_after_submit_last_order_train.rename(
    columns={'index': 'userid', 0: 'actionType234_after_submit_last_order'}, inplace=True)
actionType234_after_submit_last_order_train.fillna(0, inplace=True)
actionType234_after_submit_last_order_train.to_csv(
    feature_file_path + 'actionType234_after_submit_last_order_train.csv', index=False)

# test
history_action_merge_by_time_test = pd.read_csv(
    project_path + 'preprocess/middle_file/history_action_merge_by_time_test.csv')
orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")

userid_actionType234_num_dict_test = defaultdict(lambda: defaultdict(lambda: 0))
userid_set_submitted_order_test = set(orderHistory_test['userid'])
for userid in userid_set_submitted_order_test:
    last_orderTime_index = history_action_merge_by_time_test[
        (history_action_merge_by_time_test['userid'] == userid) & (
                history_action_merge_by_time_test['actionTime_zero_orderTime_one'] == 1)].index.values.max()
    last_actionTime_index = history_action_merge_by_time_test[
        history_action_merge_by_time_test['userid'] == userid].index.values.max()

    last_range_df = history_action_merge_by_time_test.iloc[last_orderTime_index + 1:last_actionTime_index + 1, :]

    for actionType in last_range_df['actionType'].values:
        userid_actionType234_num_dict_test[userid][actionType] += 1

userid_num_of_actionType234_dict_test = defaultdict(lambda: 0)
for userid in userid_actionType234_num_dict_test.keys():
    Type234_num = 0
    Type234_num += (userid_actionType234_num_dict_test[userid][2] + \
                    userid_actionType234_num_dict_test[userid][3] + \
                    userid_actionType234_num_dict_test[userid][4])
    userid_num_of_actionType234_dict_test[userid] = Type234_num

# print(userid_num_of_actionType234_dict_test)
actionType234_after_submit_last_order_test = pd.DataFrame.from_dict(userid_num_of_actionType234_dict_test,
                                                                    orient='index')
actionType234_after_submit_last_order_test.reset_index(inplace=True)
actionType234_after_submit_last_order_test.rename(
    columns={'index': 'userid', 0: 'actionType234_after_submit_last_order'}, inplace=True)
actionType234_after_submit_last_order_test.fillna(0, inplace=True)
actionType234_after_submit_last_order_test.to_csv(
    feature_file_path + 'actionType234_after_submit_last_order_test.csv', index=False)

# 特征 actionType_num_after_submit_last_order，表示用户在提交最后一次订单后各Type次数。
print('constructing feature actionType_num_after_submit_last_order')
history_action_merge_by_time_train = pd.read_csv(
    project_path + 'preprocess/middle_file/history_action_merge_by_time_train.csv')
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")

userid_actionType_num_dict_trian = defaultdict(lambda: defaultdict(lambda: 0))
userid_set_submitted_order_train = set(orderHistory_train['userid'])
for userid in userid_set_submitted_order_train:
    last_orderTime_index = history_action_merge_by_time_train[
        (history_action_merge_by_time_train['userid'] == userid) & (
                history_action_merge_by_time_train['actionTime_zero_orderTime_one'] == 1)].index.values.max()
    last_actionTime_index = history_action_merge_by_time_train[
        history_action_merge_by_time_train['userid'] == userid].index.values.max()

    last_range_df = history_action_merge_by_time_train.iloc[last_orderTime_index + 1:last_actionTime_index + 1, :]

    for actionType in last_range_df['actionType'].values:
        userid_actionType_num_dict_trian[userid][actionType] += 1

userid_num_of_actionTypeX_dict_train = defaultdict(lambda: [])
for userid in userid_actionType_num_dict_trian.keys():
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][1])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][2])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][3])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][4])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][5])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][6])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][7])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][8])
    userid_num_of_actionTypeX_dict_train[userid].append(userid_actionType_num_dict_trian[userid][9])

# print(userid_num_of_actionType234_dict_train)
actionType_num_after_submit_last_order_train = pd.DataFrame.from_dict(userid_num_of_actionTypeX_dict_train,
                                                                      orient='index')
actionType_num_after_submit_last_order_train.reset_index(inplace=True)
actionType_num_after_submit_last_order_train.rename(
    columns={'index': 'userid', 0: 'actionType1_num_after_submit_last_order',
             0: 'actionType1_num_after_submit_last_order', 1: 'actionType2_num_after_submit_last_order',
             2: 'actionType3_num_after_submit_last_order', 3: 'actionType4_num_after_submit_last_order',
             4: 'actionType5_num_after_submit_last_order', 5: 'actionType6_num_after_submit_last_order',
             6: 'actionType7_num_after_submit_last_order', 7: 'actionType8_num_after_submit_last_order',
             8: 'actionType9_num_after_submit_last_order'}, inplace=True)
actionType_num_after_submit_last_order_train.fillna(0, inplace=True)
actionType_num_after_submit_last_order_train.to_csv(
    feature_file_path + 'actionType_num_after_submit_last_order_train.csv', index=False)

# test
history_action_merge_by_time_test = pd.read_csv(
    project_path + 'preprocess/middle_file/history_action_merge_by_time_test.csv')
orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")

userid_actionType_num_dict_test = defaultdict(lambda: defaultdict(lambda: 0))
userid_set_submitted_order_test = set(orderHistory_test['userid'])
for userid in userid_set_submitted_order_test:
    last_orderTime_index = history_action_merge_by_time_test[
        (history_action_merge_by_time_test['userid'] == userid) & (
                history_action_merge_by_time_test['actionTime_zero_orderTime_one'] == 1)].index.values.max()
    last_actionTime_index = history_action_merge_by_time_test[
        history_action_merge_by_time_test['userid'] == userid].index.values.max()

    last_range_df = history_action_merge_by_time_test.iloc[last_orderTime_index + 1:last_actionTime_index + 1, :]

    for actionType in last_range_df['actionType'].values:
        userid_actionType_num_dict_test[userid][actionType] += 1

userid_num_of_actionTypeX_dict_test = defaultdict(lambda: [])
for userid in userid_actionType_num_dict_test.keys():
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][1])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][2])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][3])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][4])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][5])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][6])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][7])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][8])
    userid_num_of_actionTypeX_dict_test[userid].append(userid_actionType_num_dict_test[userid][9])

# print(userid_num_of_actionType234_dict_test)
actionType_num_after_submit_last_order_test = pd.DataFrame.from_dict(userid_num_of_actionTypeX_dict_test,
                                                                     orient='index')
actionType_num_after_submit_last_order_test.reset_index(inplace=True)
actionType_num_after_submit_last_order_test.rename(
    columns={'index': 'userid', 0: 'actionType1_num_after_submit_last_order',
             0: 'actionType1_num_after_submit_last_order', 1: 'actionType2_num_after_submit_last_order',
             2: 'actionType3_num_after_submit_last_order', 3: 'actionType4_num_after_submit_last_order',
             4: 'actionType5_num_after_submit_last_order', 5: 'actionType6_num_after_submit_last_order',
             6: 'actionType7_num_after_submit_last_order', 7: 'actionType8_num_after_submit_last_order',
             8: 'actionType9_num_after_submit_last_order'}, inplace=True)
actionType_num_after_submit_last_order_test.fillna(0, inplace=True)
actionType_num_after_submit_last_order_test.to_csv(
    feature_file_path + 'actionType_num_after_submit_last_order_test.csv', index=False)
# 特征 tags_num and keywords_num，表示用户评论数据中tags和keywords数量。
print('constructing feature tags_num and keywords_num')
userComment_train = pd.read_csv(traindata_path + "userComment_train.csv")
userComment_train = userComment_train.fillna(0)
userid_tags_keywords_num_train = open(feature_file_path + 'userid_tags_keywords_num_train.csv', 'w')
userid_tags_keywords_num_train.write('userid,tages_num,keywords_num')
for index in userComment_train.index:
    write_userid = userComment_train.iloc[index, 0]
    if (userComment_train.iloc[index, 3] != 0):
        write_tages_num = len(userComment_train.iloc[index, 3].split('|'))
    else:
        write_tages_num = 0
    if (userComment_train.iloc[index, 4] != 0):
        write_keywords_num = len(userComment_train.iloc[index, 4].split(',')) + 1
    else:
        write_keywords_num = 0
    userid_tags_keywords_num_train.write('\n')
    userid_tags_keywords_num_train.write(str(write_userid) + ',' + str(write_tages_num) + ',' + str(write_keywords_num))
userid_tags_keywords_num_train.close()
userid_tags_keywords_num_train = pd.read_csv(feature_file_path + 'userid_tags_keywords_num_train.csv')
userid_tags_keywords_num_train.fillna(0, inplace=True)
userid_tags_keywords_num_train.to_csv(feature_file_path + 'userid_tags_keywords_num_train.csv', index=False)

userComment_test = pd.read_csv(testdata_path + "userComment_test.csv")
userComment_test = userComment_test.fillna(0)
userid_tags_keywords_num_test = open(feature_file_path + 'userid_tags_keywords_num_test.csv', 'w')
userid_tags_keywords_num_test.write('userid,tages_num,keywords_num')
for index in userComment_test.index:
    write_userid = userComment_test.iloc[index, 0]
    if (userComment_test.iloc[index, 3] != 0):
        write_tages_num = len(userComment_test.iloc[index, 3].split('|'))
    else:
        write_tages_num = 0
    if (userComment_test.iloc[index, 4] != 0):
        write_keywords_num = len(userComment_test.iloc[index, 4].split(',')) + 1
    else:
        write_keywords_num = 0
    userid_tags_keywords_num_test.write('\n')
    userid_tags_keywords_num_test.write(str(write_userid) + ',' + str(write_tages_num) + ',' + str(write_keywords_num))
userid_tags_keywords_num_test.close()
userid_tags_keywords_num_test = pd.read_csv(feature_file_path + 'userid_tags_keywords_num_test.csv')
userid_tags_keywords_num_test.fillna(0, inplace=True)
userid_tags_keywords_num_test.to_csv(feature_file_path + 'userid_tags_keywords_num_test.csv', index=False)

# 特征 TypeX_num_by_day等9个特征+action记录中出现的天数，按日期分，用户平均每天的各Type数量
action_train = pd.read_csv(traindata_path + "action_train.csv")
action_test = pd.read_csv(testdata_path + "action_test.csv")

action_train['actionTime'] = action_train['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
action_test['actionTime'] = action_test['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))


def func(df):
    global num_label
    global num_count
    num_count += 1
    if (num_count % 1000 == 0):
        print(num_count)
    if (num_label != 0):
        actionTime_arr = df['actionTime'].unique()
        day_num1_list = []
        day_num2_list = []
        day_num3_list = []
        day_num4_list = []
        day_num5_list = []
        day_num6_list = []
        day_num7_list = []
        day_num8_list = []
        day_num9_list = []
        for time in actionTime_arr:
            actionType_arr = df[df['actionTime'] == time]['actionType']
            num1 = 0
            num2 = 0
            num3 = 0
            num4 = 0
            num5 = 0
            num6 = 0
            num7 = 0
            num8 = 0
            num9 = 0
            for type in actionType_arr:
                if (type == 1):
                    num1 += 1
                if (type == 2):
                    num2 += 1
                if (type == 3):
                    num3 += 1
                if (type == 4):
                    num4 += 1
                if (type == 5):
                    num5 += 1
                if (type == 6):
                    num6 += 1
                if (type == 7):
                    num7 += 1
                if (type == 8):
                    num8 += 1
                if (type == 9):
                    num9 += 1
            day_num1_list.append(num1)
            day_num2_list.append(num2)
            day_num3_list.append(num3)
            day_num4_list.append(num4)
            day_num5_list.append(num5)
            day_num6_list.append(num6)
            day_num7_list.append(num7)
            day_num8_list.append(num8)
            day_num9_list.append(num9)
        day_num1_arr = np.array(day_num1_list)
        day_num2_arr = np.array(day_num2_list)
        day_num3_arr = np.array(day_num3_list)
        day_num4_arr = np.array(day_num4_list)
        day_num5_arr = np.array(day_num5_list)
        day_num6_arr = np.array(day_num6_list)
        day_num7_arr = np.array(day_num7_list)
        day_num8_arr = np.array(day_num8_list)
        day_num9_arr = np.array(day_num9_list)
        df['Type1_num_by_day'] = round(np.sum(day_num1_arr) / len(day_num1_arr), 2)
        df['Type2_num_by_day'] = round(np.sum(day_num2_arr) / len(day_num2_arr), 2)
        df['Type3_num_by_day'] = round(np.sum(day_num3_arr) / len(day_num3_arr), 2)
        df['Type4_num_by_day'] = round(np.sum(day_num4_arr) / len(day_num4_arr), 2)
        df['Type5_num_by_day'] = round(np.sum(day_num5_arr) / len(day_num5_arr), 2)
        df['Type6_num_by_day'] = round(np.sum(day_num6_arr) / len(day_num6_arr), 2)
        df['Type7_num_by_day'] = round(np.sum(day_num7_arr) / len(day_num7_arr), 2)
        df['Type8_num_by_day'] = round(np.sum(day_num8_arr) / len(day_num8_arr), 2)
        df['Type9_num_by_day'] = round(np.sum(day_num9_arr) / len(day_num9_arr), 2)
        df['days'] = len(actionTime_arr)
    num_label += 1
    gc.collect()
    return df


action_train['Type1_num_by_day'] = 0
action_train['Type2_num_by_day'] = 0
action_train['Type3_num_by_day'] = 0
action_train['Type4_num_by_day'] = 0
action_train['Type5_num_by_day'] = 0
action_train['Type6_num_by_day'] = 0
action_train['Type7_num_by_day'] = 0
action_train['Type8_num_by_day'] = 0
action_train['Type9_num_by_day'] = 0
action_train['days'] = 0
num_label = 0
num_count = 0
print('\n-----------------------------\n共有：' + str(len(action_train['userid'].unique())))
action_train = action_train.groupby(['userid']).apply(func)
del action_train['actionType']
del action_train['actionTime']
action_train.drop_duplicates(inplace=True)
action_train.reset_index(drop=True, inplace=True)
action_train.to_csv(feature_file_path + 'browse_product_num_by_day_train.csv', index=False)

action_test['Type1_num_by_day'] = 0
action_test['Type2_num_by_day'] = 0
action_test['Type3_num_by_day'] = 0
action_test['Type4_num_by_day'] = 0
action_test['Type5_num_by_day'] = 0
action_test['Type6_num_by_day'] = 0
action_test['Type7_num_by_day'] = 0
action_test['Type8_num_by_day'] = 0
action_test['Type9_num_by_day'] = 0
action_test['days'] = 0
num_label = 0
num_count = 0
print('\n-----------------------------\n共有：' + str(len(action_test['userid'].unique())))
action_test = action_test.groupby(['userid']).apply(func)
del action_test['actionType']
del action_test['actionTime']
action_test.drop_duplicates(inplace=True)
action_test.reset_index(drop=True, inplace=True)
action_test.to_csv(feature_file_path + 'browse_product_num_by_day_test.csv', index=False)

# 特征TypeX_last1_day等九个特征,最后一天TypeX的次数
action_train = pd.read_csv(traindata_path + "action_train.csv")
action_test = pd.read_csv(testdata_path + "action_test.csv")

action_train['actionTime'] = action_train['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
action_test['actionTime'] = action_test['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))


def func(df):
    global num_label
    global num_count
    num_count += 1
    if (num_count % 1000 == 0):
        print(num_count)
    if (num_label != 0):
        actionTime_arr = df['actionTime'].unique()
        time = actionTime_arr[-1]
        # print('time: ' + str(time))
        actionType_arr = df[df['actionTime'] == time]['actionType']
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        num5 = 0
        num6 = 0
        num7 = 0
        num8 = 0
        num9 = 0
        for type in actionType_arr:
            if (type == 1):
                num1 += 1
            if (type == 2):
                num2 += 1
            if (type == 3):
                num3 += 1
            if (type == 4):
                num4 += 1
            if (type == 5):
                num5 += 1
            if (type == 6):
                num6 += 1
            if (type == 7):
                num7 += 1
            if (type == 8):
                num8 += 1
            if (type == 9):
                num9 += 1
        df['Type1_last1_day'] = num1
        df['Type2_last1_day'] = num2
        df['Type3_last1_day'] = num3
        df['Type4_last1_day'] = num4
        df['Type5_last1_day'] = num5
        df['Type6_last1_day'] = num6
        df['Type7_last1_day'] = num7
        df['Type8_last1_day'] = num8
        df['Type9_last1_day'] = num9
    num_label += 1
    gc.collect()
    return df


action_train['Type1_last1_day'] = 0
action_train['Type2_last1_day'] = 0
action_train['Type3_last1_day'] = 0
action_train['Type4_last1_day'] = 0
action_train['Type5_last1_day'] = 0
action_train['Type6_last1_day'] = 0
action_train['Type7_last1_day'] = 0
action_train['Type8_last1_day'] = 0
action_train['Type9_last1_day'] = 0

num_label = 0
num_count = 0
print('\n-----------------------------\n共有：' + str(len(action_train['userid'].unique())))
action_train = action_train.groupby(['userid']).apply(func)
del action_train['actionType']
del action_train['actionTime']
action_train.drop_duplicates(inplace=True)
action_train.reset_index(drop=True, inplace=True)
action_train.to_csv(feature_file_path + 'TypeX_last1_day_train.csv', index=False)

action_test['Type1_last1_day'] = 0
action_test['Type2_last1_day'] = 0
action_test['Type3_last1_day'] = 0
action_test['Type4_last1_day'] = 0
action_test['Type5_last1_day'] = 0
action_test['Type6_last1_day'] = 0
action_test['Type7_last1_day'] = 0
action_test['Type8_last1_day'] = 0
action_test['Type9_last1_day'] = 0
num_label = 0
num_count = 0
print('\n-----------------------------\n共有：' + str(len(action_test['userid'].unique())))
action_test = action_test.groupby(['userid']).apply(func)
del action_test['actionType']
del action_test['actionTime']
action_test.drop_duplicates(inplace=True)
action_test.reset_index(drop=True, inplace=True)
action_test.to_csv(feature_file_path + 'TypeX_last1_day_test.csv', index=False)

# 特征TypeX_last3_day等九个特征,最后三天TypeX的次数
action_train = pd.read_csv(traindata_path + "action_train.csv")
action_test = pd.read_csv(testdata_path + "action_test.csv")

action_train['actionTime'] = action_train['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
action_test['actionTime'] = action_test['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))


def func(df):
    global num_label
    global num_count
    num_count += 1
    if (num_count % 1000 == 0):
        print(num_count)
    if (num_label != 0):
        print(df['userid'].values[0])
        actionTime_arr = df['actionTime'].unique()
        print('actionTime_arr: ')
        print(actionTime_arr)
        # print('time: ' + str(time))
        if (len(actionTime_arr) > 2):
            time_arr = actionTime_arr[-3:]
        else:
            time_arr = actionTime_arr
        print('time_arr: ')
        print(time_arr)
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        num5 = 0
        num6 = 0
        num7 = 0
        num8 = 0
        num9 = 0
        for time in time_arr:
            actionType_arr = df[df['actionTime'] == time]['actionType']
            for type in actionType_arr:
                if (type == 1):
                    num1 += 1
                if (type == 2):
                    num2 += 1
                if (type == 3):
                    num3 += 1
                if (type == 4):
                    num4 += 1
                if (type == 5):
                    num5 += 1
                if (type == 6):
                    num6 += 1
                if (type == 7):
                    num7 += 1
                if (type == 8):
                    num8 += 1
                if (type == 9):
                    num9 += 1
        df['Type1_last3_day'] = num1
        df['Type2_last3_day'] = num2
        df['Type3_last3_day'] = num3
        df['Type4_last3_day'] = num4
        df['Type5_last3_day'] = num5
        df['Type6_last3_day'] = num6
        df['Type7_last3_day'] = num7
        df['Type8_last3_day'] = num8
        df['Type9_last3_day'] = num9
    num_label += 1
    gc.collect()
    return df


action_train['Type1_last3_day'] = 0
action_train['Type2_last3_day'] = 0
action_train['Type3_last3_day'] = 0
action_train['Type4_last3_day'] = 0
action_train['Type5_last3_day'] = 0
action_train['Type6_last3_day'] = 0
action_train['Type7_last3_day'] = 0
action_train['Type8_last3_day'] = 0
action_train['Type9_last3_day'] = 0

num_label = 0
num_count = 0
print('\n-----------------------------\n共有：' + str(len(action_train['userid'].unique())))
# action_train = action_train.head(1000)
action_train = action_train.groupby(['userid']).apply(func)
del action_train['actionType']
del action_train['actionTime']
action_train.drop_duplicates(inplace=True)
action_train.reset_index(drop=True, inplace=True)
action_train.to_csv(feature_file_path + 'TypeX_last3_day_train.csv', index=False)

action_test['Type1_last3_day'] = 0
action_test['Type2_last3_day'] = 0
action_test['Type3_last3_day'] = 0
action_test['Type4_last3_day'] = 0
action_test['Type5_last3_day'] = 0
action_test['Type6_last3_day'] = 0
action_test['Type7_last3_day'] = 0
action_test['Type8_last3_day'] = 0
action_test['Type9_last3_day'] = 0

num_label = 0
num_count = 0
print('\n-----------------------------\n共有：' + str(len(action_test['userid'].unique())))
# action_test = action_test.head(1000)
action_test = action_test.groupby(['userid']).apply(func)
del action_test['actionType']
del action_test['actionTime']
action_test.drop_duplicates(inplace=True)
action_test.reset_index(drop=True, inplace=True)
action_test.to_csv(feature_file_path + 'TypeX_last3_day_test.csv', index=False)

# 特征 order_day_mean下单天数跨度的均值，day_after_lastorder最后一次订单后浏览记录天数跨度
# 例如用户记录为从10-23开始浏览，到10-25完成下单，10-26浏览，10-27下单，10-28到10-31都是在浏览，则3天，2天为一次下单天数跨度，均值为2.5天
# 最后一次订单后浏览记录为4天，所以day_after_lastorder=4.
action_train = pd.read_csv(traindata_path + 'action_train.csv')
orderHistory_train = pd.read_csv(traindata_path + 'orderHistory_train.csv')


# orderHistory_train用户的下单顺序有的不是按照时间先后排列的，需要先进行排序。
def fun_history(df):
    time_list = df['orderTime'].values
    time_list = sorted(time_list, key=int)
    df['orderTime'] = time_list
    return df


orderHistory_train = orderHistory_train.groupby('userid').apply(fun_history)
action_train['actionTime'] = action_train['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
orderHistory_train['orderTime'] = orderHistory_train['orderTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))

userid_history_arr = orderHistory_train['userid'].unique()
# 得到{userid:[mean,day]}字典，
# for userid in userid_history_arr:
#    actionTime_arr=action_train[action_train['userid']==userid]['actionTime'].unique()
#    找到orderTime在actionTime_arr中的索引，将actionTime_arr分裂成len(orderTime)+1段，前两段均值为所求的mean,后一段值为day
# def insert_index()
userid_mean_day = defaultdict(lambda: [])
num_train = 0
for userid in userid_history_arr:
    num_train += 1
    if (num_train % 200 == 0):
        print(num_train)
    # for userid in [100000013471]:
    actionTime_arr = action_train[action_train['userid'] == userid]['actionTime'].unique()
    orderTime_arr = orderHistory_train[orderHistory_train['userid'] == userid]['orderTime'].unique()
    # for i in range(len(orderTime_arr)):
    index_a = 0
    index_b = 0
    order_day = []
    # 重构结构，之前的代码思路太乱了
    # 现在将orderTime插入进actionTime,这样就不会出现np.where()为空的情况了
    actionTime_list = actionTime_arr.tolist()
    orderTime_list = orderTime_arr.tolist()
    action_order_Timemerge = []
    action_order_Timemerge.extend(actionTime_list)
    action_order_Timemerge.extend(orderTime_list)
    action_order_Timemerge = list(set(action_order_Timemerge))
    action_order_Timemerge = sorted(action_order_Timemerge, key=str.lower)

    action_order_Timemerge_arr = np.asarray(action_order_Timemerge)
    for i in range(len(orderTime_arr)):
        # print('------------------')
        # print('i= ' + str(i))
        # print('orderTime_arr[' + str(i) + "]: " + orderTime_arr[i])
        index_a = index_b if (i == 0 or orderTime_arr[i] >= actionTime_arr[-1]) else index_b + 1
        # print('index_a: ' + str(index_a))
        # if (len(np.where(action_order_Timemerge_arr == orderTime_arr[i])[0]) != 0):
        index_b = np.where(action_order_Timemerge_arr == orderTime_arr[i])[0][0]
        # print('index_b: ' + str(index_b))
        day_b = datetime.datetime.strptime(action_order_Timemerge_arr[index_b], "%Y-%m-%d").date()
        # print('day_b: ' + str(day_b))
        day_a = datetime.datetime.strptime(action_order_Timemerge_arr[index_a], "%Y-%m-%d").date()
        # print('day_a: ' + str(day_a))
        order_day.append((day_b - day_a).days + 1)

    # print(order_day)
    mean = round(np.mean(np.asarray(order_day)), 2)
    userid_mean_day[userid].append(mean)

    if (index_b == len(action_order_Timemerge_arr) - 1):
        userid_mean_day[userid].append(0)
    else:
        index_b += 1
        day = (datetime.datetime.strptime(action_order_Timemerge_arr[-1],
                                          "%Y-%m-%d").date() - datetime.datetime.strptime(
            action_order_Timemerge_arr[index_b],
            "%Y-%m-%d").date()).days + 1
        userid_mean_day[userid].append(day)
    # print('userid_mean_day:   ')
    # print(userid_mean_day)
df = pd.DataFrame.from_dict(userid_mean_day, orient='index')
df.reset_index(inplace=True)
df.rename(columns={'index': 'userid', 0: 'order_day_mean', 1: 'day_after_lastorder'}, inplace=True)
df.to_csv(feature_file_path + 'order_day_mean_train.csv', index=False)

# test

action_test = pd.read_csv(testdata_path + 'action_test.csv')
orderHistory_test = pd.read_csv(testdata_path + 'orderHistory_test.csv')


# orderHistory_test用户的下单顺序有的不是按照时间先后排列的，需要先进行排序。
def fun_history(df):
    time_list = df['orderTime'].values
    time_list = sorted(time_list, key=int)
    df['orderTime'] = time_list
    return df


orderHistory_test = orderHistory_test.groupby('userid').apply(fun_history)
action_test['actionTime'] = action_test['actionTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
orderHistory_test['orderTime'] = orderHistory_test['orderTime'].apply(
    lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))

userid_history_arr = orderHistory_test['userid'].unique()
# 得到{userid:[mean,day]}字典，
# for userid in userid_history_arr:
#    actionTime_arr=action_test[action_test['userid']==userid]['actionTime'].unique()
#    找到orderTime在actionTime_arr中的索引，将actionTime_arr分裂成len(orderTime)+1段，前两段均值为所求的mean,后一段值为day
# def insert_index()
userid_mean_day = defaultdict(lambda: [])
num_test = 0
for userid in userid_history_arr:
    num_test += 1
    if (num_test % 200 == 0):
        print(num_test)
    # for userid in [100000013471]:
    # print('\n################\n')
    # print(userid)
    actionTime_arr = action_test[action_test['userid'] == userid]['actionTime'].unique()
    # print(type(actionTime_arr))
    orderTime_arr = orderHistory_test[orderHistory_test['userid'] == userid]['orderTime'].unique()
    # print(orderTime_arr)
    # for i in range(len(orderTime_arr)):
    index_a = 0
    index_b = 0
    order_day = []
    # 重构结构，之前的代码思路太乱了
    # 现在将orderTime插入进actionTime,这样就不会出现np.where()为空的情况了
    actionTime_list = actionTime_arr.tolist()
    orderTime_list = orderTime_arr.tolist()
    action_order_Timemerge = []
    action_order_Timemerge.extend(actionTime_list)
    action_order_Timemerge.extend(orderTime_list)
    action_order_Timemerge = list(set(action_order_Timemerge))
    action_order_Timemerge = sorted(action_order_Timemerge, key=str.lower)

    action_order_Timemerge_arr = np.asarray(action_order_Timemerge)
    for i in range(len(orderTime_arr)):
        # print('------------------')
        # print('i= ' + str(i))
        # print('orderTime_arr[' + str(i) + "]: " + orderTime_arr[i])
        index_a = index_b if (i == 0 or orderTime_arr[i] >= actionTime_arr[-1]) else index_b + 1
        # print('index_a: ' + str(index_a))
        # if (len(np.where(action_order_Timemerge_arr == orderTime_arr[i])[0]) != 0):
        index_b = np.where(action_order_Timemerge_arr == orderTime_arr[i])[0][0]
        # print('index_b: ' + str(index_b))
        day_b = datetime.datetime.strptime(action_order_Timemerge_arr[index_b], "%Y-%m-%d").date()
        # print('day_b: ' + str(day_b))
        day_a = datetime.datetime.strptime(action_order_Timemerge_arr[index_a], "%Y-%m-%d").date()
        # print('day_a: ' + str(day_a))
        order_day.append((day_b - day_a).days + 1)

    # print(order_day)
    mean = round(np.mean(np.asarray(order_day)), 2)
    userid_mean_day[userid].append(mean)

    if (index_b == len(action_order_Timemerge_arr) - 1):
        userid_mean_day[userid].append(0)
    else:
        index_b += 1
        day = (datetime.datetime.strptime(action_order_Timemerge_arr[-1],
                                          "%Y-%m-%d").date() - datetime.datetime.strptime(
            action_order_Timemerge_arr[index_b],
            "%Y-%m-%d").date()).days + 1
        userid_mean_day[userid].append(day)
    # print('userid_mean_day:   ')
    # print(userid_mean_day)
df = pd.DataFrame.from_dict(userid_mean_day, orient='index')
df.reset_index(inplace=True)
df.rename(columns={'index': 'userid', 0: 'order_day_mean', 1: 'day_after_lastorder'}, inplace=True)
df.to_csv(feature_file_path + 'order_day_mean_test.csv', index=False)

# 特征 to_firstorder_timestamp 、to_lastorder_timestamp 用户第一次和最近一次订单距离actionTime最后的时间
print('to_Xorder_timestamp...')
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderHistory_train = pd.read_csv(traindata_path + 'orderHistory_train.csv')


def func(df):
    global num
    if (num != 0):
        userid = df['userid'].values[0]
        print(userid)
        orderTime_arr = df['orderTime'].values
        actionTime_arr = action_train[action_train['userid'] == userid]['actionTime'].values
        to_lastorder_timestamp_value = actionTime_arr[-1] - orderTime_arr[-1]
        to_firstorder_timestamp_value = actionTime_arr[-1] - orderTime_arr[0]
        if (to_lastorder_timestamp_value < 0):
            to_lastorder_timestamp_value = 0
        if (to_firstorder_timestamp_value <= 0):
            to_firstorder_timestamp_value = 0
        to_Xorder_timestamp.write('\n')
        to_Xorder_timestamp.write(
            str(userid) + ',' + str(to_firstorder_timestamp_value) + ',' + str(to_lastorder_timestamp_value))
    num += 1
    return df


num = 0
to_Xorder_timestamp = open(feature_file_path + 'to_Xorder_timestamp_train.csv', 'w')
to_Xorder_timestamp.write('userid' + ',' + 'to_firstorder_timestamp' + ',' + 'to_lastorder_timestamp')
orderHistory_train.groupby('userid').apply(lambda X: func(X))
to_Xorder_timestamp.close()

action_test = pd.read_csv(testdata_path + "action_test.csv")
orderHistory_test = pd.read_csv(testdata_path + 'orderHistory_test.csv')


def func(df):
    global num
    if (num != 0):
        userid = df['userid'].values[0]
        orderTime_arr = df['orderTime'].values
        actionTime_arr = action_test[action_test['userid'] == userid]['actionTime'].values
        to_lastorder_timestamp_value = actionTime_arr[-1] - orderTime_arr[-1]
        to_firstorder_timestamp_value = actionTime_arr[-1] - orderTime_arr[0]
        if (to_lastorder_timestamp_value < 0):
            to_lastorder_timestamp_value = 0
        if (to_firstorder_timestamp_value <= 0):
            to_firstorder_timestamp_value = 0
        to_Xorder_timestamp.write('\n')
        to_Xorder_timestamp.write(
            str(userid) + ',' + str(to_firstorder_timestamp_value) + ',' + str(to_lastorder_timestamp_value))
    num += 1
    return df


num = 0
to_Xorder_timestamp = open(feature_file_path + 'to_Xorder_timestamp_test.csv', 'w')
to_Xorder_timestamp.write('userid' + ',' + 'to_firstorder_timestamp' + ',' + 'to_lastorder_timestamp')
orderHistory_test.groupby('userid').apply(lambda X: func(X))
to_Xorder_timestamp.close()


# 特征 考虑订单时间间隔order_time_interval，用户的actionTime和orderTime交叉情况有5种，如下面代码：
def func(df):
    global num1
    global num2
    global num3
    global num4
    global num5
    userid = df['userid'].values[0]
    actionTime = action_train[action_train['userid'] == userid]['actionTime'].values
    orderTime = orderHistory_train[orderHistory_train['userid'] == userid]['orderTime'].values
    if (orderTime[0] > actionTime[-1]):
        num1 += 1
    elif (orderTime[0] < actionTime[-1] and orderTime[-1] > actionTime[-1]):
        num2 += 1
    elif (orderTime[-1] <= actionTime[-1] and orderTime[0] >= actionTime[0]):
        num3 += 1
    elif (orderTime[-1] > actionTime[0] and orderTime[0] < actionTime[0]):
        num4 += 1
    elif (orderTime[-1] < actionTime[0]):
        num5 += 1
    return df


num1 = 0
num2 = 0
num3 = 0
num4 = 0
num5 = 0
action_train = pd.read_csv(traindata_path + "action_train.csv")
orderHistory_train = pd.read_csv(traindata_path + "orderHistory_train.csv")

orderHistory_train.groupby('userid').apply(lambda X: func(X))
print(num1)
print(num2)
print(num3)
print(num4)
print(num5)
# 284
# 16
# 10196
# 139
# 261


action_train = pd.read_csv(traindata_path + 'action_train.csv')
orderHistory_train = pd.read_csv(traindata_path + 'orderHistory_train.csv')
action_test = pd.read_csv(testdata_path + 'action_test.csv')
orderHistory_test = pd.read_csv(testdata_path + 'orderHistory_test.csv')


def func(df):
    global num
    if (num != 0):
        userid = df['userid'].values[0]
        print(userid)
        actionTime = action_train[action_train['userid'] == userid]['actionTime'].values
        orderTime = orderHistory_train[orderHistory_train['userid'] == userid]['orderTime'].values
        if (orderTime[0] > actionTime[-1] or orderTime[-1] < actionTime[0]):
            time_interval = []
            time_interval.append(orderTime[0] - actionTime[0])
            for i in range(1, len(orderTime)):
                time_interval.append(orderTime[i] - orderTime[i - 1])
            userid_order_time_interval_dict[userid] = time_interval
        elif (orderTime[0] < actionTime[-1] and orderTime[-1] > actionTime[-1]):
            time_interval = []
            orderTime_list = orderTime.tolist()
            action_order_Timemerge = []
            action_order_Timemerge.extend(orderTime_list)
            action_order_Timemerge.append(actionTime[-1])
            action_order_Timemerge = sorted(action_order_Timemerge, key=int)
            for i in range(1, len(action_order_Timemerge)):
                time_interval.append(action_order_Timemerge[i] - action_order_Timemerge[i - 1])
            userid_order_time_interval_dict[userid] = time_interval
        elif (orderTime[-1] <= actionTime[-1] and orderTime[0] >= actionTime[0]):
            time_interval = []
            time_interval.append(orderTime[0] - actionTime[0])
            for i in range(1, len(orderTime)):
                time_interval.append(orderTime[i] - orderTime[i - 1])
            time_interval.append(actionTime[-1] - orderTime[-1])
            userid_order_time_interval_dict[userid] = time_interval
        elif (orderTime[-1] > actionTime[0] and orderTime[0] < actionTime[0]):
            time_interval = []
            orderTime_list = orderTime.tolist()
            action_order_Timemerge = []
            action_order_Timemerge.extend(orderTime_list)
            action_order_Timemerge.append(actionTime[0])
            action_order_Timemerge.append(actionTime[-1])
            action_order_Timemerge = sorted(action_order_Timemerge, key=int)
            for i in range(1, len(action_order_Timemerge)):
                time_interval.append(action_order_Timemerge[i] - action_order_Timemerge[i - 1])
            userid_order_time_interval_dict[userid] = time_interval
        # elif (orderTime[-1] < actionTime[0]):
    num += 1
    return df


num = 0
userid_order_time_interval_dict = defaultdict(lambda: [])
orderHistory_train.groupby('userid').apply(lambda X: func(X))
order_time_interval_file = open(feature_file_path + 'order_time_interval_train.csv', 'w')
order_time_interval_file.write(
    'userid,order_time_interval_firstvale,order_time_interval_lastvale1,order_time_interval_lastvale2,order_time_interval_lastvale3,order_time_interval_mean,order_time_interval_var,order_time_interval_min,order_time_interval_max')
for userid in userid_order_time_interval_dict.keys():
    order_time_interval_file.write('\n')
    order_time_interval_arr = np.array(userid_order_time_interval_dict[userid])
    firstvale = order_time_interval_arr[0]
    lastvale1 = order_time_interval_arr[-1]
    lastvale2 = order_time_interval_arr[-2] if (len(order_time_interval_arr) > 1) else np.nan
    lastvale3 = order_time_interval_arr[-3] if (len(order_time_interval_arr) > 2) else np.nan
    vale_mean = np.mean(order_time_interval_arr)
    vale_var = np.var(order_time_interval_arr)
    vale_min = np.var(order_time_interval_arr)
    vale_max = np.max(order_time_interval_arr)
    order_time_interval_file.write(
        str(userid) + ',' + str(firstvale) + ',' + str(lastvale1) + ',' + str(lastvale2) + ',' + str(
            lastvale3) + ',' + str(vale_mean) + ',' + str(vale_var) + ',' + str(vale_min) + ',' + str(vale_max))
order_time_interval_file.close()


def func(df):
    global num
    if (num != 0):
        userid = df['userid'].values[0]
        actionTime = action_test[action_test['userid'] == userid]['actionTime'].values
        orderTime = orderHistory_test[orderHistory_test['userid'] == userid]['orderTime'].values
        if (orderTime[0] > actionTime[-1] or orderTime[-1] < actionTime[0]):
            time_interval = []
            time_interval.append(orderTime[0] - actionTime[0])
            for i in range(1, len(orderTime)):
                time_interval.append(orderTime[i] - orderTime[i - 1])
            userid_order_time_interval_dict[userid] = time_interval
        elif (orderTime[0] < actionTime[-1] and orderTime[-1] > actionTime[-1]):
            time_interval = []
            orderTime_list = orderTime.tolist()
            action_order_Timemerge = []
            action_order_Timemerge.extend(orderTime_list)
            action_order_Timemerge.append(actionTime[-1])
            action_order_Timemerge = sorted(action_order_Timemerge, key=int)
            for i in range(1, len(action_order_Timemerge)):
                time_interval.append(action_order_Timemerge[i] - action_order_Timemerge[i - 1])
            userid_order_time_interval_dict[userid] = time_interval
        elif (orderTime[-1] <= actionTime[-1] and orderTime[0] >= actionTime[0]):
            time_interval = []
            time_interval.append(orderTime[0] - actionTime[0])
            for i in range(1, len(orderTime)):
                time_interval.append(orderTime[i] - orderTime[i - 1])
            time_interval.append(actionTime[-1] - orderTime[-1])
            userid_order_time_interval_dict[userid] = time_interval
        elif (orderTime[-1] > actionTime[0] and orderTime[0] < actionTime[0]):
            time_interval = []
            orderTime_list = orderTime.tolist()
            action_order_Timemerge = []
            action_order_Timemerge.extend(orderTime_list)
            action_order_Timemerge.append(actionTime[0])
            action_order_Timemerge.append(actionTime[-1])
            action_order_Timemerge = sorted(action_order_Timemerge, key=int)
            for i in range(1, len(action_order_Timemerge)):
                time_interval.append(action_order_Timemerge[i] - action_order_Timemerge[i - 1])
            userid_order_time_interval_dict[userid] = time_interval
    num += 1
    return df


num = 0
userid_order_time_interval_dict = defaultdict(lambda: [])
orderHistory_test.groupby('userid').apply(lambda X: func(X))
order_time_interval_file = open(feature_file_path + 'order_time_interval_test.csv', 'w')
order_time_interval_file.write(
    'userid,order_time_interval_firstvale,order_time_interval_lastvale1,order_time_interval_lastvale2,order_time_interval_lastvale3,order_time_interval_mean,order_time_interval_var,order_time_interval_min,order_time_interval_max')
for userid in userid_order_time_interval_dict.keys():
    order_time_interval_file.write('\n')
    order_time_interval_arr = np.array(userid_order_time_interval_dict[userid])
    firstvale = order_time_interval_arr[0]
    lastvale1 = order_time_interval_arr[-1]
    lastvale2 = order_time_interval_arr[-2] if (len(order_time_interval_arr) > 1) else np.nan
    lastvale3 = order_time_interval_arr[-3] if (len(order_time_interval_arr) > 2) else np.nan
    vale_mean = np.mean(order_time_interval_arr)
    vale_var = np.var(order_time_interval_arr)
    vale_min = np.var(order_time_interval_arr)
    vale_max = np.max(order_time_interval_arr)
    order_time_interval_file.write(
        str(userid) + ',' + str(firstvale) + ',' + str(lastvale1) + ',' + str(lastvale2) + ',' + str(
            lastvale3) + ',' + str(vale_mean) + ',' + str(vale_var) + ',' + str(vale_min) + ',' + str(vale_max))
order_time_interval_file.close()
#
# action中各Type的时间间隔均值等信息，action_TypeX_interval_time_info
action_train = pd.read_csv(traindata_path + 'action_train.csv')
# action_train = action_train.head(1000)
action_TypeX_interval_time_info = open(feature_file_path + 'action_TypeX_interval_time_info_train.csv', 'w')
action_TypeX_interval_time_info.write(
    'userid,' +
    'Type1_interval_time_firstvale,Type1_interval_time_lastvale,Type1_interval_time_lastvale2,Type1_interval_time_lastvale3,Type1_interval_time_mean,Type1_interval_time_var,Type1_interval_time_min,Type1_interval_time_max,' +
    'Type2_interval_time_firstvale,Type2_interval_time_lastvale,Type2_interval_time_lastvale2,Type2_interval_time_lastvale3,Type2_interval_time_mean,Type2_interval_time_var,Type2_interval_time_min,Type2_interval_time_max,' +
    'Type3_interval_time_firstvale,Type3_interval_time_lastvale,Type3_interval_time_lastvale2,Type3_interval_time_lastvale3,Type3_interval_time_mean,Type3_interval_time_var,Type3_interval_time_min,Type3_interval_time_max,' +
    'Type4_interval_time_firstvale,Type4_interval_time_lastvale,Type4_interval_time_lastvale2,Type4_interval_time_lastvale3,Type4_interval_time_mean,Type4_interval_time_var,Type4_interval_time_min,Type4_interval_time_max,' +
    'Type5_interval_time_firstvale,Type5_interval_time_lastvale,Type5_interval_time_lastvale2,Type5_interval_time_lastvale3,Type5_interval_time_mean,Type5_interval_time_var,Type5_interval_time_min,Type5_interval_time_max,' +
    'Type6_interval_time_firstvale,Type6_interval_time_lastvale,Type6_interval_time_lastvale2,Type6_interval_time_lastvale3,Type6_interval_time_mean,Type6_interval_time_var,Type6_interval_time_min,Type6_interval_time_max,' +
    'Type7_interval_time_firstvale,Type7_interval_time_lastvale,Type7_interval_time_lastvale2,Type7_interval_time_lastvale3,Type7_interval_time_mean,Type7_interval_time_var,Type7_interval_time_min,Type7_interval_time_max,' +
    'Type8_interval_time_firstvale,Type8_interval_time_lastvale,Type8_interval_time_lastvale2,Type8_interval_time_lastvale3,Type8_interval_time_mean,Type8_interval_time_var,Type8_interval_time_min,Type8_interval_time_max,' +
    'Type9_interval_time_firstvale,Type9_interval_time_lastvale,Type9_interval_time_lastvale2,Type9_interval_time_lastvale3,Type9_interval_time_mean,Type9_interval_time_var,Type9_interval_time_min,Type9_interval_time_max')


def func(df):
    global num
    global count
    if (num != 0):
        if (count % 500 == 0):
            print(count)
        userid = df['userid'].values[0]
        if (1 in df['actionType'].values):
            Type1_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type1_interval_time_list = []
            for i in range(1, len(Type1_actionTime_arr)):
                Type1_interval_time_list.append(Type1_actionTime_arr[i] - Type1_actionTime_arr[i - 1])
            Type1_interval_time_arr = np.array(Type1_interval_time_list)
            Type1_interval_time_firstvale = Type1_interval_time_arr[0] if (len(Type1_interval_time_arr) > 0) else np.nan
            Type1_interval_time_lastvale = Type1_interval_time_arr[-1] if (len(Type1_interval_time_arr) > 0) else np.nan
            Type1_interval_time_lastvale2 = Type1_interval_time_arr[-2] if (
                    len(Type1_interval_time_arr) > 1) else np.nan
            Type1_interval_time_lastvale3 = Type1_interval_time_arr[-3] if (
                    len(Type1_interval_time_arr) > 2) else np.nan
            Type1_interval_time_mean = round(np.mean(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 1) else np.nan
            Type1_interval_time_var = round(np.var(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 1) else np.nan
            Type1_interval_time_min = round(np.min(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 0) else np.nan
            Type1_interval_time_max = round(np.max(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 0) else np.nan
        else:
            Type1_interval_time_firstvale = np.nan
            Type1_interval_time_lastvale = np.nan
            Type1_interval_time_lastvale2 = np.nan
            Type1_interval_time_lastvale3 = np.nan
            Type1_interval_time_mean = np.nan
            Type1_interval_time_var = np.nan
            Type1_interval_time_min = np.nan
            Type1_interval_time_max = np.nan
        if (2 in df['actionType'].values):
            Type2_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type2_interval_time_list = []
            for i in range(1, len(Type2_actionTime_arr)):
                Type2_interval_time_list.append(Type2_actionTime_arr[i] - Type2_actionTime_arr[i - 1])
            Type2_interval_time_arr = np.array(Type2_interval_time_list)
            Type2_interval_time_firstvale = Type2_interval_time_arr[0] if (len(Type2_interval_time_arr) > 0) else np.nan
            Type2_interval_time_lastvale = Type2_interval_time_arr[-1] if (len(Type2_interval_time_arr) > 0) else np.nan
            Type2_interval_time_lastvale2 = Type2_interval_time_arr[-2] if (
                    len(Type2_interval_time_arr) > 1) else np.nan
            Type2_interval_time_lastvale3 = Type2_interval_time_arr[-3] if (
                    len(Type2_interval_time_arr) > 2) else np.nan
            Type2_interval_time_mean = round(np.mean(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 1) else np.nan
            Type2_interval_time_var = round(np.var(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 1) else np.nan
            Type2_interval_time_min = round(np.min(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 0) else np.nan
            Type2_interval_time_max = round(np.max(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 0) else np.nan
        else:
            Type2_interval_time_firstvale = np.nan
            Type2_interval_time_lastvale = np.nan
            Type2_interval_time_lastvale2 = np.nan
            Type2_interval_time_lastvale3 = np.nan
            Type2_interval_time_mean = np.nan
            Type2_interval_time_var = np.nan
            Type2_interval_time_min = np.nan
            Type2_interval_time_max = np.nan
        if (3 in df['actionType'].values):
            Type3_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type3_interval_time_list = []
            for i in range(1, len(Type3_actionTime_arr)):
                Type3_interval_time_list.append(Type3_actionTime_arr[i] - Type3_actionTime_arr[i - 1])
            Type3_interval_time_arr = np.array(Type3_interval_time_list)
            Type3_interval_time_firstvale = Type3_interval_time_arr[0] if (len(Type3_interval_time_arr) > 0) else np.nan
            Type3_interval_time_lastvale = Type3_interval_time_arr[-1] if (len(Type3_interval_time_arr) > 0) else np.nan
            Type3_interval_time_lastvale2 = Type3_interval_time_arr[-2] if (
                    len(Type3_interval_time_arr) > 1) else np.nan
            Type3_interval_time_lastvale3 = Type3_interval_time_arr[-3] if (
                    len(Type3_interval_time_arr) > 2) else np.nan
            Type3_interval_time_mean = round(np.mean(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 1) else np.nan
            Type3_interval_time_var = round(np.var(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 1) else np.nan
            Type3_interval_time_min = round(np.min(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 0) else np.nan
            Type3_interval_time_max = round(np.max(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 0) else np.nan
        else:
            Type3_interval_time_firstvale = np.nan
            Type3_interval_time_lastvale = np.nan
            Type3_interval_time_lastvale2 = np.nan
            Type3_interval_time_lastvale3 = np.nan
            Type3_interval_time_mean = np.nan
            Type3_interval_time_var = np.nan
            Type3_interval_time_min = np.nan
            Type3_interval_time_max = np.nan
        if (4 in df['actionType'].values):
            Type4_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type4_interval_time_list = []
            for i in range(1, len(Type4_actionTime_arr)):
                Type4_interval_time_list.append(Type4_actionTime_arr[i] - Type4_actionTime_arr[i - 1])
            Type4_interval_time_arr = np.array(Type4_interval_time_list)
            Type4_interval_time_firstvale = Type4_interval_time_arr[0] if (len(Type4_interval_time_arr) > 0) else np.nan
            Type4_interval_time_lastvale = Type4_interval_time_arr[-1] if (len(Type4_interval_time_arr) > 0) else np.nan
            Type4_interval_time_lastvale2 = Type4_interval_time_arr[-2] if (
                    len(Type4_interval_time_arr) > 1) else np.nan
            Type4_interval_time_lastvale3 = Type4_interval_time_arr[-3] if (
                    len(Type4_interval_time_arr) > 2) else np.nan
            Type4_interval_time_mean = round(np.mean(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 1) else np.nan
            Type4_interval_time_var = round(np.var(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 1) else np.nan
            Type4_interval_time_min = round(np.min(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 0) else np.nan
            Type4_interval_time_max = round(np.max(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 0) else np.nan
        else:
            Type4_interval_time_firstvale = np.nan
            Type4_interval_time_lastvale = np.nan
            Type4_interval_time_lastvale2 = np.nan
            Type4_interval_time_lastvale3 = np.nan
            Type4_interval_time_mean = np.nan
            Type4_interval_time_var = np.nan
            Type4_interval_time_min = np.nan
            Type4_interval_time_max = np.nan
        if (5 in df['actionType'].values):
            Type5_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type5_interval_time_list = []
            for i in range(1, len(Type5_actionTime_arr)):
                Type5_interval_time_list.append(Type5_actionTime_arr[i] - Type5_actionTime_arr[i - 1])
            Type5_interval_time_arr = np.array(Type5_interval_time_list)
            Type5_interval_time_firstvale = Type5_interval_time_arr[0] if (len(Type5_interval_time_arr) > 0) else np.nan
            Type5_interval_time_lastvale = Type5_interval_time_arr[-1] if (len(Type5_interval_time_arr) > 0) else np.nan
            Type5_interval_time_lastvale2 = Type5_interval_time_arr[-2] if (
                    len(Type5_interval_time_arr) > 1) else np.nan
            Type5_interval_time_lastvale3 = Type5_interval_time_arr[-3] if (
                    len(Type5_interval_time_arr) > 2) else np.nan
            Type5_interval_time_mean = round(np.mean(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 1) else np.nan
            Type5_interval_time_var = round(np.var(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 1) else np.nan
            Type5_interval_time_min = round(np.min(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 0) else np.nan
            Type5_interval_time_max = round(np.max(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 0) else np.nan
        else:
            Type5_interval_time_firstvale = np.nan
            Type5_interval_time_lastvale = np.nan
            Type5_interval_time_lastvale2 = np.nan
            Type5_interval_time_lastvale3 = np.nan
            Type5_interval_time_mean = np.nan
            Type5_interval_time_var = np.nan
            Type5_interval_time_min = np.nan
            Type5_interval_time_max = np.nan
        if (6 in df['actionType'].values):
            Type6_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type6_interval_time_list = []
            for i in range(1, len(Type6_actionTime_arr)):
                Type6_interval_time_list.append(Type6_actionTime_arr[i] - Type6_actionTime_arr[i - 1])
            Type6_interval_time_arr = np.array(Type6_interval_time_list)
            Type6_interval_time_firstvale = Type6_interval_time_arr[0] if (len(Type6_interval_time_arr) > 0) else np.nan
            Type6_interval_time_lastvale = Type6_interval_time_arr[-1] if (len(Type6_interval_time_arr) > 0) else np.nan
            Type6_interval_time_lastvale2 = Type6_interval_time_arr[-2] if (
                    len(Type6_interval_time_arr) > 1) else np.nan
            Type6_interval_time_lastvale3 = Type6_interval_time_arr[-3] if (
                    len(Type6_interval_time_arr) > 2) else np.nan
            Type6_interval_time_mean = round(np.mean(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 1) else np.nan
            Type6_interval_time_var = round(np.var(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 1) else np.nan
            Type6_interval_time_min = round(np.min(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 0) else np.nan
            Type6_interval_time_max = round(np.max(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 0) else np.nan
        else:
            Type6_interval_time_firstvale = np.nan
            Type6_interval_time_lastvale = np.nan
            Type6_interval_time_lastvale2 = np.nan
            Type6_interval_time_lastvale3 = np.nan
            Type6_interval_time_mean = np.nan
            Type6_interval_time_var = np.nan
            Type6_interval_time_min = np.nan
            Type6_interval_time_max = np.nan
        if (7 in df['actionType'].values):
            Type7_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type7_interval_time_list = []
            for i in range(1, len(Type7_actionTime_arr)):
                Type7_interval_time_list.append(Type7_actionTime_arr[i] - Type7_actionTime_arr[i - 1])
            Type7_interval_time_arr = np.array(Type7_interval_time_list)
            Type7_interval_time_firstvale = Type7_interval_time_arr[0] if (len(Type7_interval_time_arr) > 0) else np.nan
            Type7_interval_time_lastvale = Type7_interval_time_arr[-1] if (len(Type7_interval_time_arr) > 0) else np.nan
            Type7_interval_time_lastvale2 = Type7_interval_time_arr[-2] if (
                    len(Type7_interval_time_arr) > 1) else np.nan
            Type7_interval_time_lastvale3 = Type7_interval_time_arr[-3] if (
                    len(Type7_interval_time_arr) > 2) else np.nan
            Type7_interval_time_mean = round(np.mean(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 1) else np.nan
            Type7_interval_time_var = round(np.var(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 1) else np.nan
            Type7_interval_time_min = round(np.min(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 0) else np.nan
            Type7_interval_time_max = round(np.max(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 0) else np.nan
        else:
            Type7_interval_time_firstvale = np.nan
            Type7_interval_time_lastvale = np.nan
            Type7_interval_time_lastvale2 = np.nan
            Type7_interval_time_lastvale3 = np.nan
            Type7_interval_time_mean = np.nan
            Type7_interval_time_var = np.nan
            Type7_interval_time_min = np.nan
            Type7_interval_time_max = np.nan
        if (8 in df['actionType'].values):
            Type8_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type8_interval_time_list = []
            for i in range(1, len(Type8_actionTime_arr)):
                Type8_interval_time_list.append(Type8_actionTime_arr[i] - Type8_actionTime_arr[i - 1])
            Type8_interval_time_arr = np.array(Type8_interval_time_list)
            Type8_interval_time_firstvale = Type8_interval_time_arr[0] if (len(Type8_interval_time_arr) > 0) else np.nan
            Type8_interval_time_lastvale = Type8_interval_time_arr[-1] if (len(Type8_interval_time_arr) > 0) else np.nan
            Type8_interval_time_lastvale2 = Type8_interval_time_arr[-2] if (
                    len(Type8_interval_time_arr) > 1) else np.nan
            Type8_interval_time_lastvale3 = Type8_interval_time_arr[-3] if (
                    len(Type8_interval_time_arr) > 2) else np.nan
            Type8_interval_time_mean = round(np.mean(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 1) else np.nan
            Type8_interval_time_var = round(np.var(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 1) else np.nan
            Type8_interval_time_min = round(np.min(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 0) else np.nan
            Type8_interval_time_max = round(np.max(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 0) else np.nan
        else:
            Type8_interval_time_firstvale = np.nan
            Type8_interval_time_lastvale = np.nan
            Type8_interval_time_lastvale2 = np.nan
            Type8_interval_time_lastvale3 = np.nan
            Type8_interval_time_mean = np.nan
            Type8_interval_time_var = np.nan
            Type8_interval_time_min = np.nan
            Type8_interval_time_max = np.nan
        if (9 in df['actionType'].values):
            Type9_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type9_interval_time_list = []
            for i in range(1, len(Type9_actionTime_arr)):
                Type9_interval_time_list.append(Type9_actionTime_arr[i] - Type9_actionTime_arr[i - 1])
            Type9_interval_time_arr = np.array(Type9_interval_time_list)
            Type9_interval_time_firstvale = Type9_interval_time_arr[0] if (len(Type9_interval_time_arr) > 0) else np.nan
            Type9_interval_time_lastvale = Type9_interval_time_arr[-1] if (len(Type9_interval_time_arr) > 0) else np.nan
            Type9_interval_time_lastvale2 = Type9_interval_time_arr[-2] if (
                    len(Type9_interval_time_arr) > 1) else np.nan
            Type9_interval_time_lastvale3 = Type9_interval_time_arr[-3] if (
                    len(Type9_interval_time_arr) > 2) else np.nan
            Type9_interval_time_mean = round(np.mean(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 1) else np.nan
            Type9_interval_time_var = round(np.var(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 1) else np.nan
            Type9_interval_time_min = round(np.min(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 0) else np.nan
            Type9_interval_time_max = round(np.max(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 0) else np.nan
        else:
            Type9_interval_time_firstvale = np.nan
            Type9_interval_time_lastvale = np.nan
            Type9_interval_time_lastvale2 = np.nan
            Type9_interval_time_lastvale3 = np.nan
            Type9_interval_time_mean = np.nan
            Type9_interval_time_var = np.nan
            Type9_interval_time_min = np.nan
            Type9_interval_time_max = np.nan

        action_TypeX_interval_time_info.write('\n')
        action_TypeX_interval_time_info.write(str(userid) + ',' +
                                              str(Type1_interval_time_firstvale) + ',' + str(
            Type1_interval_time_lastvale) + ',' + str(Type1_interval_time_lastvale2) + ',' + str(
            Type1_interval_time_lastvale3) + ',' + str(Type1_interval_time_mean) + ',' + str(
            Type1_interval_time_var) + ',' + str(Type1_interval_time_min) + ',' + str(Type1_interval_time_max) + ',' +
                                              str(Type2_interval_time_firstvale) + ',' + str(
            Type2_interval_time_lastvale) + ',' + str(Type2_interval_time_lastvale2) + ',' + str(
            Type2_interval_time_lastvale3) + ',' + str(Type2_interval_time_mean) + ',' + str(
            Type2_interval_time_var) + ',' + str(Type2_interval_time_min) + ',' + str(Type2_interval_time_max) + ',' +
                                              str(
                                                  Type3_interval_time_firstvale) + ',' + str(
            Type3_interval_time_lastvale) + ',' + str(Type3_interval_time_lastvale2) + ',' + str(
            Type3_interval_time_lastvale3) + ',' + str(Type3_interval_time_mean) + ',' + str(
            Type3_interval_time_var) + ',' + str(Type3_interval_time_min) + ',' + str(Type3_interval_time_max) + ',' +
                                              str(
                                                  Type4_interval_time_firstvale) + ',' + str(
            Type4_interval_time_lastvale) + ',' + str(Type4_interval_time_lastvale2) + ',' + str(
            Type4_interval_time_lastvale3) + ',' + str(Type4_interval_time_mean) + ',' + str(
            Type4_interval_time_var) + ',' + str(Type4_interval_time_min) + ',' + str(Type4_interval_time_max) + ',' +
                                              str(
                                                  Type5_interval_time_firstvale) + ',' + str(
            Type5_interval_time_lastvale) + ',' + str(Type5_interval_time_lastvale2) + ',' + str(
            Type5_interval_time_lastvale3) + ',' + str(Type5_interval_time_mean) + ',' + str(
            Type5_interval_time_var) + ',' + str(Type5_interval_time_min) + ',' + str(Type5_interval_time_max) + ',' +
                                              str(
                                                  Type6_interval_time_firstvale) + ',' + str(
            Type6_interval_time_lastvale) + ',' + str(Type6_interval_time_lastvale2) + ',' + str(
            Type6_interval_time_lastvale3) + ',' + str(Type6_interval_time_mean) + ',' + str(
            Type6_interval_time_var) + ',' + str(Type6_interval_time_min) + ',' + str(Type6_interval_time_max) + ',' +
                                              str(
                                                  Type7_interval_time_firstvale) + ',' + str(
            Type7_interval_time_lastvale) + ',' + str(Type7_interval_time_lastvale2) + ',' + str(
            Type7_interval_time_lastvale3) + ',' + str(Type7_interval_time_mean) + ',' + str(
            Type7_interval_time_var) + ',' + str(Type7_interval_time_min) + ',' + str(Type7_interval_time_max) + ',' +
                                              str(
                                                  Type8_interval_time_firstvale) + ',' + str(
            Type8_interval_time_lastvale) + ',' + str(Type8_interval_time_lastvale2) + ',' + str(
            Type8_interval_time_lastvale3) + ',' + str(Type8_interval_time_mean) + ',' + str(
            Type8_interval_time_var) + ',' + str(Type8_interval_time_min) + ',' + str(Type8_interval_time_max) + ',' +
                                              str(
                                                  Type9_interval_time_firstvale) + ',' + str(
            Type9_interval_time_lastvale) + ',' + str(Type9_interval_time_lastvale2) + ',' + str(
            Type9_interval_time_lastvale3) + ',' + str(Type9_interval_time_mean) + ',' + str(
            Type9_interval_time_var) + ',' + str(Type9_interval_time_min) + ',' + str(Type9_interval_time_max))

        gc.collect()
    count += 1
    num += 1
    return df


num = 0
count = 0
action_train.groupby(['userid']).apply(func)
action_TypeX_interval_time_info.close()

# # test
# action中各Type的时间间隔均值等信息，action_TypeX_interval_time_info
action_test = pd.read_csv(testdata_path + 'action_test.csv')
# action_test = action_test.head(1000)
action_TypeX_interval_time_info = open(feature_file_path + 'action_TypeX_interval_time_info_test.csv', 'w')
action_TypeX_interval_time_info.write(
    'userid,' +
    'Type1_interval_time_firstvale,Type1_interval_time_lastvale,Type1_interval_time_lastvale2,Type1_interval_time_lastvale3,Type1_interval_time_mean,Type1_interval_time_var,Type1_interval_time_min,Type1_interval_time_max,' +
    'Type2_interval_time_firstvale,Type2_interval_time_lastvale,Type2_interval_time_lastvale2,Type2_interval_time_lastvale3,Type2_interval_time_mean,Type2_interval_time_var,Type2_interval_time_min,Type2_interval_time_max,' +
    'Type3_interval_time_firstvale,Type3_interval_time_lastvale,Type3_interval_time_lastvale2,Type3_interval_time_lastvale3,Type3_interval_time_mean,Type3_interval_time_var,Type3_interval_time_min,Type3_interval_time_max,' +
    'Type4_interval_time_firstvale,Type4_interval_time_lastvale,Type4_interval_time_lastvale2,Type4_interval_time_lastvale3,Type4_interval_time_mean,Type4_interval_time_var,Type4_interval_time_min,Type4_interval_time_max,' +
    'Type5_interval_time_firstvale,Type5_interval_time_lastvale,Type5_interval_time_lastvale2,Type5_interval_time_lastvale3,Type5_interval_time_mean,Type5_interval_time_var,Type5_interval_time_min,Type5_interval_time_max,' +
    'Type6_interval_time_firstvale,Type6_interval_time_lastvale,Type6_interval_time_lastvale2,Type6_interval_time_lastvale3,Type6_interval_time_mean,Type6_interval_time_var,Type6_interval_time_min,Type6_interval_time_max,' +
    'Type7_interval_time_firstvale,Type7_interval_time_lastvale,Type7_interval_time_lastvale2,Type7_interval_time_lastvale3,Type7_interval_time_mean,Type7_interval_time_var,Type7_interval_time_min,Type7_interval_time_max,' +
    'Type8_interval_time_firstvale,Type8_interval_time_lastvale,Type8_interval_time_lastvale2,Type8_interval_time_lastvale3,Type8_interval_time_mean,Type8_interval_time_var,Type8_interval_time_min,Type8_interval_time_max,' +
    'Type9_interval_time_firstvale,Type9_interval_time_lastvale,Type9_interval_time_lastvale2,Type9_interval_time_lastvale3,Type9_interval_time_mean,Type9_interval_time_var,Type9_interval_time_min,Type9_interval_time_max')


def func(df):
    global num
    global count
    if (num != 0):
        if (count % 500 == 0):
            print(count)
        userid = df['userid'].values[0]
        if (1 in df['actionType'].values):
            Type1_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type1_interval_time_list = []
            for i in range(1, len(Type1_actionTime_arr)):
                Type1_interval_time_list.append(Type1_actionTime_arr[i] - Type1_actionTime_arr[i - 1])
            Type1_interval_time_arr = np.array(Type1_interval_time_list)
            Type1_interval_time_firstvale = Type1_interval_time_arr[0] if (len(Type1_interval_time_arr) > 0) else np.nan
            Type1_interval_time_lastvale = Type1_interval_time_arr[-1] if (len(Type1_interval_time_arr) > 0) else np.nan
            Type1_interval_time_lastvale2 = Type1_interval_time_arr[-2] if (
                    len(Type1_interval_time_arr) > 1) else np.nan
            Type1_interval_time_lastvale3 = Type1_interval_time_arr[-3] if (
                    len(Type1_interval_time_arr) > 2) else np.nan
            Type1_interval_time_mean = round(np.mean(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 1) else np.nan
            Type1_interval_time_var = round(np.var(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 1) else np.nan
            Type1_interval_time_min = round(np.min(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 0) else np.nan
            Type1_interval_time_max = round(np.max(Type1_interval_time_arr), 2) if (
                    len(Type1_interval_time_arr) > 0) else np.nan
        else:
            Type1_interval_time_firstvale = np.nan
            Type1_interval_time_lastvale = np.nan
            Type1_interval_time_lastvale2 = np.nan
            Type1_interval_time_lastvale3 = np.nan
            Type1_interval_time_mean = np.nan
            Type1_interval_time_var = np.nan
            Type1_interval_time_min = np.nan
            Type1_interval_time_max = np.nan
        if (2 in df['actionType'].values):
            Type2_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type2_interval_time_list = []
            for i in range(1, len(Type2_actionTime_arr)):
                Type2_interval_time_list.append(Type2_actionTime_arr[i] - Type2_actionTime_arr[i - 1])
            Type2_interval_time_arr = np.array(Type2_interval_time_list)
            Type2_interval_time_firstvale = Type2_interval_time_arr[0] if (len(Type2_interval_time_arr) > 0) else np.nan
            Type2_interval_time_lastvale = Type2_interval_time_arr[-1] if (len(Type2_interval_time_arr) > 0) else np.nan
            Type2_interval_time_lastvale2 = Type2_interval_time_arr[-2] if (
                    len(Type2_interval_time_arr) > 1) else np.nan
            Type2_interval_time_lastvale3 = Type2_interval_time_arr[-3] if (
                    len(Type2_interval_time_arr) > 2) else np.nan
            Type2_interval_time_mean = round(np.mean(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 1) else np.nan
            Type2_interval_time_var = round(np.var(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 1) else np.nan
            Type2_interval_time_min = round(np.min(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 0) else np.nan
            Type2_interval_time_max = round(np.max(Type2_interval_time_arr), 2) if (
                    len(Type2_interval_time_arr) > 0) else np.nan
        else:
            Type2_interval_time_firstvale = np.nan
            Type2_interval_time_lastvale = np.nan
            Type2_interval_time_lastvale2 = np.nan
            Type2_interval_time_lastvale3 = np.nan
            Type2_interval_time_mean = np.nan
            Type2_interval_time_var = np.nan
            Type2_interval_time_min = np.nan
            Type2_interval_time_max = np.nan
        if (3 in df['actionType'].values):
            Type3_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type3_interval_time_list = []
            for i in range(1, len(Type3_actionTime_arr)):
                Type3_interval_time_list.append(Type3_actionTime_arr[i] - Type3_actionTime_arr[i - 1])
            Type3_interval_time_arr = np.array(Type3_interval_time_list)
            Type3_interval_time_firstvale = Type3_interval_time_arr[0] if (len(Type3_interval_time_arr) > 0) else np.nan
            Type3_interval_time_lastvale = Type3_interval_time_arr[-1] if (len(Type3_interval_time_arr) > 0) else np.nan
            Type3_interval_time_lastvale2 = Type3_interval_time_arr[-2] if (
                    len(Type3_interval_time_arr) > 1) else np.nan
            Type3_interval_time_lastvale3 = Type3_interval_time_arr[-3] if (
                    len(Type3_interval_time_arr) > 2) else np.nan
            Type3_interval_time_mean = round(np.mean(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 1) else np.nan
            Type3_interval_time_var = round(np.var(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 1) else np.nan
            Type3_interval_time_min = round(np.min(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 0) else np.nan
            Type3_interval_time_max = round(np.max(Type3_interval_time_arr), 2) if (
                    len(Type3_interval_time_arr) > 0) else np.nan
        else:
            Type3_interval_time_firstvale = np.nan
            Type3_interval_time_lastvale = np.nan
            Type3_interval_time_lastvale2 = np.nan
            Type3_interval_time_lastvale3 = np.nan
            Type3_interval_time_mean = np.nan
            Type3_interval_time_var = np.nan
            Type3_interval_time_min = np.nan
            Type3_interval_time_max = np.nan
        if (4 in df['actionType'].values):
            Type4_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type4_interval_time_list = []
            for i in range(1, len(Type4_actionTime_arr)):
                Type4_interval_time_list.append(Type4_actionTime_arr[i] - Type4_actionTime_arr[i - 1])
            Type4_interval_time_arr = np.array(Type4_interval_time_list)
            Type4_interval_time_firstvale = Type4_interval_time_arr[0] if (len(Type4_interval_time_arr) > 0) else np.nan
            Type4_interval_time_lastvale = Type4_interval_time_arr[-1] if (len(Type4_interval_time_arr) > 0) else np.nan
            Type4_interval_time_lastvale2 = Type4_interval_time_arr[-2] if (
                    len(Type4_interval_time_arr) > 1) else np.nan
            Type4_interval_time_lastvale3 = Type4_interval_time_arr[-3] if (
                    len(Type4_interval_time_arr) > 2) else np.nan
            Type4_interval_time_mean = round(np.mean(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 1) else np.nan
            Type4_interval_time_var = round(np.var(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 1) else np.nan
            Type4_interval_time_min = round(np.min(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 0) else np.nan
            Type4_interval_time_max = round(np.max(Type4_interval_time_arr), 2) if (
                    len(Type4_interval_time_arr) > 0) else np.nan
        else:
            Type4_interval_time_firstvale = np.nan
            Type4_interval_time_lastvale = np.nan
            Type4_interval_time_lastvale2 = np.nan
            Type4_interval_time_lastvale3 = np.nan
            Type4_interval_time_mean = np.nan
            Type4_interval_time_var = np.nan
            Type4_interval_time_min = np.nan
            Type4_interval_time_max = np.nan
        if (5 in df['actionType'].values):
            Type5_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type5_interval_time_list = []
            for i in range(1, len(Type5_actionTime_arr)):
                Type5_interval_time_list.append(Type5_actionTime_arr[i] - Type5_actionTime_arr[i - 1])
            Type5_interval_time_arr = np.array(Type5_interval_time_list)
            Type5_interval_time_firstvale = Type5_interval_time_arr[0] if (len(Type5_interval_time_arr) > 0) else np.nan
            Type5_interval_time_lastvale = Type5_interval_time_arr[-1] if (len(Type5_interval_time_arr) > 0) else np.nan
            Type5_interval_time_lastvale2 = Type5_interval_time_arr[-2] if (
                    len(Type5_interval_time_arr) > 1) else np.nan
            Type5_interval_time_lastvale3 = Type5_interval_time_arr[-3] if (
                    len(Type5_interval_time_arr) > 2) else np.nan
            Type5_interval_time_mean = round(np.mean(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 1) else np.nan
            Type5_interval_time_var = round(np.var(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 1) else np.nan
            Type5_interval_time_min = round(np.min(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 0) else np.nan
            Type5_interval_time_max = round(np.max(Type5_interval_time_arr), 2) if (
                    len(Type5_interval_time_arr) > 0) else np.nan
        else:
            Type5_interval_time_firstvale = np.nan
            Type5_interval_time_lastvale = np.nan
            Type5_interval_time_lastvale2 = np.nan
            Type5_interval_time_lastvale3 = np.nan
            Type5_interval_time_mean = np.nan
            Type5_interval_time_var = np.nan
            Type5_interval_time_min = np.nan
            Type5_interval_time_max = np.nan
        if (6 in df['actionType'].values):
            Type6_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type6_interval_time_list = []
            for i in range(1, len(Type6_actionTime_arr)):
                Type6_interval_time_list.append(Type6_actionTime_arr[i] - Type6_actionTime_arr[i - 1])
            Type6_interval_time_arr = np.array(Type6_interval_time_list)
            Type6_interval_time_firstvale = Type6_interval_time_arr[0] if (len(Type6_interval_time_arr) > 0) else np.nan
            Type6_interval_time_lastvale = Type6_interval_time_arr[-1] if (len(Type6_interval_time_arr) > 0) else np.nan
            Type6_interval_time_lastvale2 = Type6_interval_time_arr[-2] if (
                    len(Type6_interval_time_arr) > 1) else np.nan
            Type6_interval_time_lastvale3 = Type6_interval_time_arr[-3] if (
                    len(Type6_interval_time_arr) > 2) else np.nan
            Type6_interval_time_mean = round(np.mean(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 1) else np.nan
            Type6_interval_time_var = round(np.var(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 1) else np.nan
            Type6_interval_time_min = round(np.min(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 0) else np.nan
            Type6_interval_time_max = round(np.max(Type6_interval_time_arr), 2) if (
                    len(Type6_interval_time_arr) > 0) else np.nan
        else:
            Type6_interval_time_firstvale = np.nan
            Type6_interval_time_lastvale = np.nan
            Type6_interval_time_lastvale2 = np.nan
            Type6_interval_time_lastvale3 = np.nan
            Type6_interval_time_mean = np.nan
            Type6_interval_time_var = np.nan
            Type6_interval_time_min = np.nan
            Type6_interval_time_max = np.nan
        if (7 in df['actionType'].values):
            Type7_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type7_interval_time_list = []
            for i in range(1, len(Type7_actionTime_arr)):
                Type7_interval_time_list.append(Type7_actionTime_arr[i] - Type7_actionTime_arr[i - 1])
            Type7_interval_time_arr = np.array(Type7_interval_time_list)
            Type7_interval_time_firstvale = Type7_interval_time_arr[0] if (len(Type7_interval_time_arr) > 0) else np.nan
            Type7_interval_time_lastvale = Type7_interval_time_arr[-1] if (len(Type7_interval_time_arr) > 0) else np.nan
            Type7_interval_time_lastvale2 = Type7_interval_time_arr[-2] if (
                    len(Type7_interval_time_arr) > 1) else np.nan
            Type7_interval_time_lastvale3 = Type7_interval_time_arr[-3] if (
                    len(Type7_interval_time_arr) > 2) else np.nan
            Type7_interval_time_mean = round(np.mean(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 1) else np.nan
            Type7_interval_time_var = round(np.var(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 1) else np.nan
            Type7_interval_time_min = round(np.min(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 0) else np.nan
            Type7_interval_time_max = round(np.max(Type7_interval_time_arr), 2) if (
                    len(Type7_interval_time_arr) > 0) else np.nan
        else:
            Type7_interval_time_firstvale = np.nan
            Type7_interval_time_lastvale = np.nan
            Type7_interval_time_lastvale2 = np.nan
            Type7_interval_time_lastvale3 = np.nan
            Type7_interval_time_mean = np.nan
            Type7_interval_time_var = np.nan
            Type7_interval_time_min = np.nan
            Type7_interval_time_max = np.nan
        if (8 in df['actionType'].values):
            Type8_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type8_interval_time_list = []
            for i in range(1, len(Type8_actionTime_arr)):
                Type8_interval_time_list.append(Type8_actionTime_arr[i] - Type8_actionTime_arr[i - 1])
            Type8_interval_time_arr = np.array(Type8_interval_time_list)
            Type8_interval_time_firstvale = Type8_interval_time_arr[0] if (len(Type8_interval_time_arr) > 0) else np.nan
            Type8_interval_time_lastvale = Type8_interval_time_arr[-1] if (len(Type8_interval_time_arr) > 0) else np.nan
            Type8_interval_time_lastvale2 = Type8_interval_time_arr[-2] if (
                    len(Type8_interval_time_arr) > 1) else np.nan
            Type8_interval_time_lastvale3 = Type8_interval_time_arr[-3] if (
                    len(Type8_interval_time_arr) > 2) else np.nan
            Type8_interval_time_mean = round(np.mean(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 1) else np.nan
            Type8_interval_time_var = round(np.var(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 1) else np.nan
            Type8_interval_time_min = round(np.min(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 0) else np.nan
            Type8_interval_time_max = round(np.max(Type8_interval_time_arr), 2) if (
                    len(Type8_interval_time_arr) > 0) else np.nan
        else:
            Type8_interval_time_firstvale = np.nan
            Type8_interval_time_lastvale = np.nan
            Type8_interval_time_lastvale2 = np.nan
            Type8_interval_time_lastvale3 = np.nan
            Type8_interval_time_mean = np.nan
            Type8_interval_time_var = np.nan
            Type8_interval_time_min = np.nan
            Type8_interval_time_max = np.nan
        if (9 in df['actionType'].values):
            Type9_actionTime_arr = df[df['actionType'] == 1]['actionTime'].values
            Type9_interval_time_list = []
            for i in range(1, len(Type9_actionTime_arr)):
                Type9_interval_time_list.append(Type9_actionTime_arr[i] - Type9_actionTime_arr[i - 1])
            Type9_interval_time_arr = np.array(Type9_interval_time_list)
            Type9_interval_time_firstvale = Type9_interval_time_arr[0] if (len(Type9_interval_time_arr) > 0) else np.nan
            Type9_interval_time_lastvale = Type9_interval_time_arr[-1] if (len(Type9_interval_time_arr) > 0) else np.nan
            Type9_interval_time_lastvale2 = Type9_interval_time_arr[-2] if (
                    len(Type9_interval_time_arr) > 1) else np.nan
            Type9_interval_time_lastvale3 = Type9_interval_time_arr[-3] if (
                    len(Type9_interval_time_arr) > 2) else np.nan
            Type9_interval_time_mean = round(np.mean(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 1) else np.nan
            Type9_interval_time_var = round(np.var(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 1) else np.nan
            Type9_interval_time_min = round(np.min(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 0) else np.nan
            Type9_interval_time_max = round(np.max(Type9_interval_time_arr), 2) if (
                    len(Type9_interval_time_arr) > 0) else np.nan
        else:
            Type9_interval_time_firstvale = np.nan
            Type9_interval_time_lastvale = np.nan
            Type9_interval_time_lastvale2 = np.nan
            Type9_interval_time_lastvale3 = np.nan
            Type9_interval_time_mean = np.nan
            Type9_interval_time_var = np.nan
            Type9_interval_time_min = np.nan
            Type9_interval_time_max = np.nan

        action_TypeX_interval_time_info.write('\n')
        action_TypeX_interval_time_info.write(str(userid) + ',' +
                                              str(Type1_interval_time_firstvale) + ',' + str(
            Type1_interval_time_lastvale) + ',' + str(Type1_interval_time_lastvale2) + ',' + str(
            Type1_interval_time_lastvale3) + ',' + str(Type1_interval_time_mean) + ',' + str(
            Type1_interval_time_var) + ',' + str(Type1_interval_time_min) + ',' + str(Type1_interval_time_max) + ',' +
                                              str(Type2_interval_time_firstvale) + ',' + str(
            Type2_interval_time_lastvale) + ',' + str(Type2_interval_time_lastvale2) + ',' + str(
            Type2_interval_time_lastvale3) + ',' + str(Type2_interval_time_mean) + ',' + str(
            Type2_interval_time_var) + ',' + str(Type2_interval_time_min) + ',' + str(Type2_interval_time_max) + ',' +
                                              str(
                                                  Type3_interval_time_firstvale) + ',' + str(
            Type3_interval_time_lastvale) + ',' + str(Type3_interval_time_lastvale2) + ',' + str(
            Type3_interval_time_lastvale3) + ',' + str(Type3_interval_time_mean) + ',' + str(
            Type3_interval_time_var) + ',' + str(Type3_interval_time_min) + ',' + str(Type3_interval_time_max) + ',' +
                                              str(
                                                  Type4_interval_time_firstvale) + ',' + str(
            Type4_interval_time_lastvale) + ',' + str(Type4_interval_time_lastvale2) + ',' + str(
            Type4_interval_time_lastvale3) + ',' + str(Type4_interval_time_mean) + ',' + str(
            Type4_interval_time_var) + ',' + str(Type4_interval_time_min) + ',' + str(Type4_interval_time_max) + ',' +
                                              str(
                                                  Type5_interval_time_firstvale) + ',' + str(
            Type5_interval_time_lastvale) + ',' + str(Type5_interval_time_lastvale2) + ',' + str(
            Type5_interval_time_lastvale3) + ',' + str(Type5_interval_time_mean) + ',' + str(
            Type5_interval_time_var) + ',' + str(Type5_interval_time_min) + ',' + str(Type5_interval_time_max) + ',' +
                                              str(
                                                  Type6_interval_time_firstvale) + ',' + str(
            Type6_interval_time_lastvale) + ',' + str(Type6_interval_time_lastvale2) + ',' + str(
            Type6_interval_time_lastvale3) + ',' + str(Type6_interval_time_mean) + ',' + str(
            Type6_interval_time_var) + ',' + str(Type6_interval_time_min) + ',' + str(Type6_interval_time_max) + ',' +
                                              str(
                                                  Type7_interval_time_firstvale) + ',' + str(
            Type7_interval_time_lastvale) + ',' + str(Type7_interval_time_lastvale2) + ',' + str(
            Type7_interval_time_lastvale3) + ',' + str(Type7_interval_time_mean) + ',' + str(
            Type7_interval_time_var) + ',' + str(Type7_interval_time_min) + ',' + str(Type7_interval_time_max) + ',' +
                                              str(
                                                  Type8_interval_time_firstvale) + ',' + str(
            Type8_interval_time_lastvale) + ',' + str(Type8_interval_time_lastvale2) + ',' + str(
            Type8_interval_time_lastvale3) + ',' + str(Type8_interval_time_mean) + ',' + str(
            Type8_interval_time_var) + ',' + str(Type8_interval_time_min) + ',' + str(Type8_interval_time_max) + ',' +
                                              str(
                                                  Type9_interval_time_firstvale) + ',' + str(
            Type9_interval_time_lastvale) + ',' + str(Type9_interval_time_lastvale2) + ',' + str(
            Type9_interval_time_lastvale3) + ',' + str(Type9_interval_time_mean) + ',' + str(
            Type9_interval_time_var) + ',' + str(Type9_interval_time_min) + ',' + str(Type9_interval_time_max))

        gc.collect()
    count += 1
    num += 1
    return df


num = 0
count = 0
action_test.groupby(['userid']).apply(func)
action_TypeX_interval_time_info.close()
