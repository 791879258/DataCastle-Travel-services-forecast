#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-29 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
traindata_base = pd.read_csv(project_path + "preprocess/traindata_base.csv")
testdata_base = pd.read_csv(project_path + "preprocess/testdata_base.csv")
traindata_output_path = project_path + "preprocess/traindata_output/"
testdata_output_path = project_path + "preprocess/testdata_output/"
feature_file_path = project_path + 'preprocess/feature_file/'
feature_from_jingsaiquan_path = feature_file_path + 'feature_from_jingsaiquan/'

###########-----------------------读取特征并与xxxdaata_base.csv合并----------------------------###########
# 用户是否购买过精品旅游服务
ever_purchase_orderType1_train = pd.read_csv(feature_file_path + "ever_purchase_orderType1_train.csv")
ever_purchase_orderType1_test = pd.read_csv(feature_file_path + "ever_purchase_orderType1_test.csv")

traindata_output = pd.merge(traindata_base, ever_purchase_orderType1_train, how='left', on='userid')
testdata_output = pd.merge(testdata_base, ever_purchase_orderType1_test, how='left', on='userid')
traindata_output.fillna(0, inplace=True)
testdata_output.fillna(0, inplace=True)
print(traindata_output.shape)
print(testdata_output.shape)
# # 用户是否在APP购买过服务（包括普通服务和精品服务）
# user_purchase_orderType01_times_train = pd.read_csv(feature_file_path + 'user_purchase_orderType01_times_train.csv')
# user_purchase_orderType01_times_test = pd.read_csv(feature_file_path + 'user_purchase_orderType01_times_test.csv')
# traindata_output = pd.merge(traindata_output, user_purchase_orderType01_times_train, how='left', on='userid')
# testdata_output = pd.merge(testdata_output, user_purchase_orderType01_times_test, how='left', on='userid')
#
# # 用户的评分中是否存在>=3的评分
# if_exist_rate_largerthan_2_train = pd.read_csv(feature_file_path + 'if_exist_rate_largerthan_2_train.csv')
# if_exist_rate_largerthan_2_test = pd.read_csv(feature_file_path + 'if_exist_rate_largerthan_2_test.csv')
# traindata_output = pd.merge(traindata_output, if_exist_rate_largerthan_2_train, how='left', on='userid')
# testdata_output = pd.merge(testdata_output, if_exist_rate_largerthan_2_test, how='left', on='userid')

# 用户的点击率
user_click_train = pd.read_csv(feature_file_path + 'user_click_train.csv')
user_click_test = pd.read_csv(feature_file_path + 'user_click_test.csv')
traindata_output = pd.merge(traindata_output, user_click_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, user_click_test, how='left', on='userid')
del traindata_output['user_open_app_num']
del testdata_output['user_open_app_num']

#
# 购买了普通服务的用户评分>2
user_orderType0_rate_largerthan_2_train = pd.read_csv(feature_file_path + 'user_orderType0_rate_largerthan_2_train.csv')
user_orderType0_rate_largerthan_2_test = pd.read_csv(feature_file_path + 'user_orderType0_rate_largerthan_2_test.csv')
traindata_output = pd.merge(traindata_output, user_orderType0_rate_largerthan_2_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, user_orderType0_rate_largerthan_2_test, how='left', on='userid')
#
# 用户在提交最后一次订单后继续浏览产品的次数
actionType234_after_submit_last_order_train = pd.read_csv(
    feature_file_path + 'actionType234_after_submit_last_order_train.csv')
actionType234_after_submit_last_order_test = pd.read_csv(
    feature_file_path + 'actionType234_after_submit_last_order_test.csv')
traindata_output = pd.merge(traindata_output, actionType234_after_submit_last_order_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, actionType234_after_submit_last_order_test, how='left', on='userid')
#
# 用户在提交最后一次订单后各Type的数量
actionType_num_after_submit_last_order_train = pd.read_csv(
    feature_file_path + 'actionType_num_after_submit_last_order_train.csv')
actionType_num_after_submit_last_order_test = pd.read_csv(
    feature_file_path + 'actionType_num_after_submit_last_order_test.csv')
traindata_output = pd.merge(traindata_output, actionType_num_after_submit_last_order_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, actionType_num_after_submit_last_order_test, how='left', on='userid')
#  用户评论数据中tags和keywords数量userid_tags_keywords_num。
userid_tags_keywords_num_train = pd.read_csv(feature_file_path + 'userid_tags_keywords_num_train.csv')
userid_tags_keywords_num_test = pd.read_csv(feature_file_path + 'userid_tags_keywords_num_test.csv')
traindata_output = pd.merge(traindata_output, userid_tags_keywords_num_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, userid_tags_keywords_num_test, how='left', on='userid')
#
# 特征 browse_product_num_by_day，按日期分
browse_product_num_by_day_train = pd.read_csv(feature_file_path + 'browse_product_num_by_day_train.csv')
browse_product_num_by_day_test = pd.read_csv(feature_file_path + 'browse_product_num_by_day_test.csv')
traindata_output = pd.merge(traindata_output, browse_product_num_by_day_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, browse_product_num_by_day_test, how='left', on='userid')
#
# 特征 TypeX_last1_day
TypeX_last1_day_train = pd.read_csv(feature_file_path + 'TypeX_last1_day_train.csv')
TypeX_last1_day_test = pd.read_csv(feature_file_path + 'TypeX_last1_day_test.csv')
traindata_output = pd.merge(traindata_output, TypeX_last1_day_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, TypeX_last1_day_test, how='left', on='userid')

# 特征 TypeX_last3_day
TypeX_last3_day_train = pd.read_csv(feature_file_path + 'TypeX_last3_day_train.csv')
TypeX_last3_day_test = pd.read_csv(feature_file_path + 'TypeX_last3_day_test.csv')
traindata_output = pd.merge(traindata_output, TypeX_last3_day_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, TypeX_last3_day_test, how='left', on='userid')
#
# 特征 order_day_mean
order_day_mean_train = pd.read_csv(feature_file_path + 'order_day_mean_train.csv')
order_day_mean_test = pd.read_csv(feature_file_path + 'order_day_mean_test.csv')
traindata_output = pd.merge(traindata_output, order_day_mean_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, order_day_mean_test, how='left', on='userid')
#
# 特征to_Xorder_timestamp
to_Xorder_timestamp_train = pd.read_csv(feature_file_path + 'to_Xorder_timestamp_train.csv')
to_Xorder_timestamp_test = pd.read_csv(feature_file_path + 'to_Xorder_timestamp_test.csv')
traindata_output = pd.merge(traindata_output, to_Xorder_timestamp_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, to_Xorder_timestamp_test, how='left', on='userid')
#
# 特征 order_time_interval_xxx
order_time_interval_train = pd.read_csv(feature_file_path + 'order_time_interval_train.csv')
order_time_interval_test = pd.read_csv(feature_file_path + 'order_time_interval_test.csv')
traindata_output = pd.merge(traindata_output, order_time_interval_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, order_time_interval_test, how='left', on='userid')

# 特征 action_TypeX_interval_time_info_XXX
action_TypeX_interval_time_info_train = pd.read_csv(feature_file_path + 'action_TypeX_interval_time_info_train.csv')
action_TypeX_interval_time_info_test = pd.read_csv(feature_file_path + 'action_TypeX_interval_time_info_test.csv')
for i in range(1, 10):
    del action_TypeX_interval_time_info_train['Type' + str(i) + '_interval_time_firstvale']
    del action_TypeX_interval_time_info_train['Type' + str(i) + '_interval_time_lastvale']
    del action_TypeX_interval_time_info_train['Type' + str(i) + '_interval_time_lastvale2']
    del action_TypeX_interval_time_info_train['Type' + str(i) + '_interval_time_lastvale3']
    del action_TypeX_interval_time_info_train['Type' + str(i) + '_interval_time_var']
    del action_TypeX_interval_time_info_train['Type' + str(i) + '_interval_time_min']
    del action_TypeX_interval_time_info_train['Type' + str(i) + '_interval_time_max']
    del action_TypeX_interval_time_info_test['Type' + str(i) + '_interval_time_firstvale']
    del action_TypeX_interval_time_info_test['Type' + str(i) + '_interval_time_lastvale']
    del action_TypeX_interval_time_info_test['Type' + str(i) + '_interval_time_lastvale2']
    del action_TypeX_interval_time_info_test['Type' + str(i) + '_interval_time_lastvale3']
    del action_TypeX_interval_time_info_test['Type' + str(i) + '_interval_time_var']
    del action_TypeX_interval_time_info_test['Type' + str(i) + '_interval_time_min']
    del action_TypeX_interval_time_info_test['Type' + str(i) + '_interval_time_max']
traindata_output = pd.merge(traindata_output, action_TypeX_interval_time_info_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, action_TypeX_interval_time_info_test, how='left', on='userid')
print(traindata_output.shape)
print(testdata_output.shape)
# fs_feature_xxx
fs_feature_train = pd.read_csv(feature_file_path + 'fs_feature_train.csv')
fs_feature_test = pd.read_csv(feature_file_path + 'fs_feature_test.csv')
traindata_output = pd.merge(traindata_output, fs_feature_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, fs_feature_test, how='left', on='userid')
print(traindata_output.shape)
print(testdata_output.shape)
# # actionTime_period_info_train
# actionTime_period_info_train = pd.read_csv(feature_file_path + 'actionTime_period_info_train.csv')
# actionTime_period_info_test = pd.read_csv(feature_file_path + 'actionTime_period_info_test.csv')
# traindata_output = pd.merge(traindata_output, actionTime_period_info_train, how='left', on='userid')
# testdata_output = pd.merge(testdata_output, actionTime_period_info_test, how='left', on='userid')
# print(traindata_output.shape)
# print(testdata_output.shape)
# 特征 lastX_time_
lastX_time_train = pd.read_csv(feature_file_path + 'lastX_time_train.csv')
lastX_time_test = pd.read_csv(feature_file_path + 'lastX_time_test.csv')
traindata_output = pd.merge(traindata_output, lastX_time_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, lastX_time_test, how='left', on='userid')
print(traindata_output.shape)
print(testdata_output.shape)
# 特征
mahout_train_feature = pd.read_csv(
    'E:\Document\Competition\DC_Mac\DC_Code\preprocess\\feature_file\\' + 'mahout_train_feature.csv')
mahout_test_feature = pd.read_csv(
    'E:\Document\Competition\DC_Mac\DC_Code\preprocess\\feature_file\\' + 'mahout_test_feature.csv')
traindata_output = pd.merge(traindata_output, mahout_train_feature, how='left', on='userid')
testdata_output = pd.merge(testdata_output, mahout_test_feature, how='left', on='userid')
print(traindata_output.shape)
print(testdata_output.shape)

# 竞赛圈的特征
clickX_rate_train = pd.read_csv(feature_from_jingsaiquan_path + 'clickX_rate_train.csv')
clickX_rate_test = pd.read_csv(feature_from_jingsaiquan_path + 'clickX_rate_test.csv')
traindata_output = pd.merge(traindata_output, clickX_rate_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, clickX_rate_test, how='left', on='userid')

########################################################################################
daoshuType_123_train = pd.read_csv(feature_from_jingsaiquan_path + 'daoshuType_123_train.csv')
daoshuType_123_test = pd.read_csv(feature_from_jingsaiquan_path + 'daoshuType_123_test.csv')
traindata_output = pd.merge(traindata_output, daoshuType_123_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, daoshuType_123_test, how='left', on='userid')

to_closest_X_infomation_train = pd.read_csv(feature_from_jingsaiquan_path + 'to_closest_X_infomation_train.csv')
to_closest_X_infomation_test = pd.read_csv(feature_from_jingsaiquan_path + 'to_closest_X_infomation_test.csv')
traindata_output = pd.merge(traindata_output, to_closest_X_infomation_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, to_closest_X_infomation_test, how='left', on='userid')

userid_time_interval_train = pd.read_csv(feature_from_jingsaiquan_path + 'userid_time_interval_train.csv')
userid_time_interval_test = pd.read_csv(feature_from_jingsaiquan_path + 'userid_time_interval_test.csv')
traindata_output = pd.merge(traindata_output, userid_time_interval_train, how='left', on='userid')
testdata_output = pd.merge(testdata_output, userid_time_interval_test, how='left', on='userid')
if (traindata_output.shape[0] != 40307):
    print('error')
###########-----------------------输出xxxdata_output.csv----------------------------###########

# traindata_output.fillna(0, inplace=True)
# testdata_output.fillna(0, inplace=True)

traindata_output.to_csv(traindata_output_path + 'traindata_output.csv', index=False)
testdata_output.to_csv(testdata_output_path + 'testdata_output.csv', index=False)
print(traindata_output.shape)
print(testdata_output.shape)
