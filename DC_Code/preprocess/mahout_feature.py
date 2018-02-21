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

# data = pd.read_csv('E:\Document\mahout_DT\data\python_file.txt', sep=" ", header=None)
# data.columns = ["userid"]
# trainfile = open('E:\Document\Competition\DC_Mac\DC_Code\preprocess\\feature_file\\' + 'mahout_train_feature.csv', 'w')
# trainfile.write('userid,'
#                 + 'suserid1,suserid1_ftype,'
#                 + 'suserid2,suserid2_ftype,'
#                 + 'suserid3,suserid3_ftype,'
#                 + 'suserid4,suserid4_ftype,'
#                 + 'suserid5,suserid5_ftype,'
#                 + 'suserid6,suserid6_ftype,'
#                 + 'suserid7,suserid7_ftype,'
#                 + 'suserid8,suserid8_ftype,'
#                 + 'suserid9,suserid9_ftype,'
#                 + 'suserid10,suserid10_ftype,'
#                 + 'suserid11,suserid11_ftype,'
#                 + 'suserid12,suserid12_ftype,'
#                 + 'suserid13,suserid13_ftype,'
#                 + 'suserid14,suserid14_ftype,'
#                 + 'suserid15,suserid15_ftype,'
#                 + 'suserid16,suserid16_ftype,'
#                 + 'suserid17,suserid17_ftype,'
#                 + 'suserid18,suserid18_ftype,'
#                 + 'suserid19,suserid19_ftype,'
#                 + 'suserid20,suserid20_ftype,'
#                 + 'suserid21,suserid21_ftype,'
#                 + 'suserid22,suserid22_ftype,'
#                 + 'suserid23,suserid23_ftype,'
#                 + 'suserid24,suserid24_ftype,'
#                 + 'suserid25,suserid25_ftype,'
#                 + 'suserid26,suserid26_ftype,'
#                 + 'suserid27,suserid27_ftype,'
#                 + 'suserid28,suserid28_ftype,'
#                 + 'suserid29,suserid29_ftype,'
#                 + 'suserid30,suserid30_ftype')
# testfile = open('E:\Document\Competition\DC_Mac\DC_Code\preprocess\\feature_file\\' + 'mahout_test_feature.csv', 'w')
# testfile.write('userid,'
#                + 'suserid1,suserid1_ftype,'
#                + 'suserid2,suserid2_ftype,'
#                + 'suserid3,suserid3_ftype,'
#                + 'suserid4,suserid4_ftype,'
#                + 'suserid5,suserid5_ftype,'
#                + 'suserid6,suserid6_ftype,'
#                + 'suserid7,suserid7_ftype,'
#                + 'suserid8,suserid8_ftype,'
#                + 'suserid9,suserid9_ftype,'
#                + 'suserid10,suserid10_ftype,'
#                + 'suserid11,suserid11_ftype,'
#                + 'suserid12,suserid12_ftype,'
#                + 'suserid13,suserid13_ftype,'
#                + 'suserid14,suserid14_ftype,'
#                + 'suserid15,suserid15_ftype,'
#                + 'suserid16,suserid16_ftype,'
#                + 'suserid17,suserid17_ftype,'
#                + 'suserid18,suserid18_ftype,'
#                + 'suserid19,suserid19_ftype,'
#                + 'suserid20,suserid20_ftype,'
#                + 'suserid21,suserid21_ftype,'
#                + 'suserid22,suserid22_ftype,'
#                + 'suserid23,suserid23_ftype,'
#                + 'suserid24,suserid24_ftype,'
#                + 'suserid25,suserid25_ftype,'
#                + 'suserid26,suserid26_ftype,'
#                + 'suserid27,suserid27_ftype,'
#                + 'suserid28,suserid28_ftype,'
#                + 'suserid29,suserid29_ftype,'
#                + 'suserid30,suserid30_ftype')
#
# userid_train_arr = orderFuture_train['userid'].values
# userid_test_arr = orderFuture_test['userid'].values
# for index, row in data.iterrows():
#     if (index % 100 == 0):
#         print(index)
#     userid_arr = row['userid'].split(',')
#     userid = int(userid_arr[0])
#     if (userid in userid_train_arr):
#         trainfile.write('\n')
#         trainfile.write(str(userid))
#         for i in range(1, 31):
#             if (int(userid_arr[i]) in userid_train_arr):
#                 ftype = orderFuture_train[orderFuture_train['userid'] == int(userid_arr[i])]['orderType'].values[0]
#                 trainfile.write(',' + userid_arr[i] + ',' + str(ftype))
#             else:
#                 trainfile.write(',' + str(np.nan) + ',' + str(np.nan))
#     else:
#         testfile.write('\n')
#         testfile.write(str(userid))
#         for i in range(1, 31):
#             if (int(userid_arr[i]) in userid_train_arr):
#                 ftype = orderFuture_train[orderFuture_train['userid'] == int(userid_arr[i])]['orderType'].values[0]
#                 testfile.write(',' + userid_arr[i] + ',' + str(ftype))
#             else:
#                 testfile.write(',' + str(np.nan) + ',' + str(np.nan))
# trainfile.close()
# testfile.close()
data = pd.read_csv('E:\Document\mahout_DT\data\python_file.txt', sep=" ", header=None)
data.columns = ["userid"]
trainfile = open('E:\Document\Competition\DC_Mac\DC_Code\preprocess\\feature_file\\' + 'mahout_train_feature.csv', 'w')
trainfile.write('userid,'
                + 'type1_rate,'
                + 'type0_rate')
testfile = open('E:\Document\Competition\DC_Mac\DC_Code\preprocess\\feature_file\\' + 'mahout_test_feature.csv', 'w')
testfile.write('userid,'
               + 'type1_rate,'
               + 'type0_rate')

userid_train_arr = orderFuture_train['userid'].values
userid_test_arr = orderFuture_test['userid'].values
for index, row in data.iterrows():
    if (index % 100 == 0):
        print(index)
    userid_arr = row['userid'].split(',')
    userid = int(userid_arr[0])
    type1_num = 0
    type0_num = 0
    if (userid in userid_train_arr):
        trainfile.write('\n')
        trainfile.write(str(userid))
        for i in range(1, 31):
            if (int(userid_arr[i]) in userid_train_arr):
                ftype = orderFuture_train[orderFuture_train['userid'] == int(userid_arr[i])]['orderType'].values[0]
                if (ftype == 0):
                    type0_num += 1
                else:
                    type1_num += 1
        num = type0_num + type1_num
        trainfile.write(',' + str(float(type0_num) / num) + ',' + str(float(type1_num) / num))

    else:
        testfile.write('\n')
        testfile.write(str(userid))
        for i in range(1, 31):
            if (int(userid_arr[i]) in userid_train_arr):
                ftype = orderFuture_train[orderFuture_train['userid'] == int(userid_arr[i])]['orderType'].values[0]
                if (ftype == 0):
                    type0_num += 1
                else:
                    type1_num += 1
        num = type0_num + type1_num
        testfile.write(',' + str(float(type0_num) / num) + ',' + str(float(type1_num) / num))
trainfile.close()
testfile.close()
