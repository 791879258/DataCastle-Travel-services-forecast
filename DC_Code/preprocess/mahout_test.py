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

# indata = input('Input:')
# arr = indata.split(',')
# userid = arr[0]
# print(str(userid) + '的future_orderType： ' + str(
#     orderFuture_train[orderFuture_train['userid'] == int(userid)]['orderType'].values[0]))
# num0 = 0
# num1 = 0
# for i in range(1, len(arr)):
#     if (orderFuture_train[orderFuture_train['userid'] == int(arr[i])]['orderType'].values[0] == 0):
#         num0 += 1
#     else:
#         num1 += 1
# print("在最相近的" + str(len(arr) - 1) + '个userid中，\nfuture_type为0有：' + str(num0) + '\nfuture_type为1有：' + str(num1))
# # print(arr)

data = pd.read_csv('E:\Document\mahout_DT\data\python_file.txt', sep=" ", header=None)
data.columns = ["a", "b", "c", "etc."]