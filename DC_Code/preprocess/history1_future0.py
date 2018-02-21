#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-03 10:16:20
# @Author  : guanglinzhou (xdzgl812@163.com)


# 统计历史订单类型为1的用户，但是未来订单为0的人数。

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

# orderHistory_train中orderType出现过1的用户,也就是订购过精品服务的用户:orderHistory_orderType_exist_one
orderHistory_orderType_exist_one = pd.DataFrame(
    columns=[['userid', 'orderid', 'orderTime', 'orderType', 'city', 'country', 'continent']])

orderHistory_orderType_exist_one_num = 0


def fun_construct_orderHistory_orderType_exist_one(df):
    global orderHistory_orderType_exist_one
    global orderHistory_orderType_exist_one_num
    if (1 in df['orderType'].values):
        if (orderHistory_orderType_exist_one_num != 0):  # 由于groupby()会传递第一组数据两次，所以跳过第一次
            orderHistory_orderType_exist_one = pd.concat([orderHistory_orderType_exist_one, df], axis=0)
    orderHistory_orderType_exist_one_num += 1


orderHistory_train.groupby('userid').apply(fun_construct_orderHistory_orderType_exist_one)
print(orderHistory_orderType_exist_one)

userid_list_of_future0 = orderFuture_train[orderFuture_train['orderType'] == 0]['userid'].values.tolist()
print('有' + str(len(set(userid_list_of_future0) & set(orderHistory_orderType_exist_one['userid'].values.tolist())))+'用户历史记录选择过精品服务，未来不选择精品服务')

