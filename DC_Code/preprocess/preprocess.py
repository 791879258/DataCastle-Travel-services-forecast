#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-25 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

################
print("start processing traindata")
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
# userProfile_train.head()
# 	userid	gender	province	age
# 0	100000000013	男	NaN	60后
# 1	100000000111	NaN	上海	NaN
# 2	100000000127	NaN	上海	NaN
# 3	100000000231	男	北京	70后
# 4	100000000379	男	北京	NaN
# userProfile_train['gender'] = userProfile_train['gender'].map({'男': "M", '女': "W"})
# userProfile_train['province'] = userProfile_train['province'].map(
#     {'上海': "shanghai", '北京': "beijing", '河南': "henan", "广东": "guangdong", "辽宁": "liaoning", "陕西": "shaanxi",
#      "浙江": "zhejiang", "四川": "sichuan", "云南": "yunnan", "黑龙江": "heilongjiang", "安徽": "anhui", "江苏": "jiangsu",
#      "重庆": "chongqing", "天津": "tianjin", "福建": "fujian", "山东": 'shandong', "湖北": "hubei", "贵州": "guizhou",
#      "湖南": "hunan", "宁夏": "ningxia", "广西": "guangxi", "吉林": 'jilin', '江西': 'jiangxi', '甘肃': 'gansu', '新疆': 'xinjiang',
#      '河北': 'hebei', '海南': 'hainan', '山西': "shanxi", '内蒙古': 'neimenggu', '青海': 'qinghai', '西藏': 'xizang'})
# userProfile_train['age'] = userProfile_train['age'].map(
#     {'60后': "60hou", '70后': "70hou", '80后': '80hou', '90后': '90hou', '00后': '00hou'})
# 将性别和省份转换为OneHot编码方式
# userProfile_train = pd.get_dummies(userProfile_train, columns=['age', 'gender'])
action_train_new = open(project_path + 'preprocess/middle_file/' + 'action_train_new.csv', 'w')
action_train_new.write(
    'userid,actionType1,actionType2,actionType3,actionType4,actionType5,actionType6,actionType7,actionType8,actionType9')
for userid in action_train['userid'].unique():
    actiontype_arr = np.array(action_train[action_train['userid'] == userid]['actionType'])
    cnt = Counter()
    for actiontype in actiontype_arr:
        cnt[actiontype] += 1
    action_train_new.write('\n')
    action_train_new.write(
        str(userid) + ',' + str(cnt[1]) + ',' + str(cnt[2]) + ',' + str(cnt[3]) + ',' + str(cnt[4]) + ',' + str(
            cnt[5]) + ',' + str(cnt[6]) + ',' + str(cnt[7]) + ',' + str(cnt[8]) + ',' + str(cnt[9]))
action_train_new.close()
action_train = pd.read_csv(project_path + 'preprocess/middle_file/' + 'action_train_new.csv')
train_data = pd.merge(action_train, userProfile_train, how='left', on="userid")
train_data = pd.merge(train_data, orderFuture_train, how='left', on="userid")
del train_data['age']
del train_data['gender']
del train_data['province']
train_data.to_csv(traindata_output_path + "traindata_base.csv", index=False)

######################
print("start processing testdata")
testdata_path = project_path + "data/test/"
testdata_output_path = project_path + "preprocess/"
action_test = pd.read_csv(testdata_path + "action_test.csv")
orderFuture_test = pd.read_csv(testdata_path + "orderFuture_test.csv")
orderHistory_test = pd.read_csv(testdata_path + "orderHistory_test.csv")
userComment_test = pd.read_csv(testdata_path + "userComment_test.csv")
userProfile_test = pd.read_csv(testdata_path + "userProfile_test.csv")

# userProfile_test.head()
# 	userid	gender	province	age
# 0	100000000013	男	NaN	60后
# 1	100000000111	NaN	上海	NaN
# 2	100000000127	NaN	上海	NaN
# 3	100000000231	男	北京	70后
# 4	100000000379	男	北京	NaN
# userProfile_test['gender'] = userProfile_test['gender'].map({'男': "M", '女': "W"})
# userProfile_test['province'] = userProfile_test['province'].map(
#     {'上海': "shanghai", '北京': "beijing", '河南': "henan", "广东": "guangdong", "辽宁": "liaoning", "陕西": "shaanxi",
#      "浙江": "zhejiang", "四川": "sichuan", "云南": "yunnan", "黑龙江": "heilongjiang", "安徽": "anhui", "江苏": "jiangsu",
#      "重庆": "chongqing", "天津": "tianjin", "福建": "fujian", "山东": 'shandong', "湖北": "hubei", "贵州": "guizhou",
#      "湖南": "hunan", "宁夏": "ningxia", "广西": "guangxi", "吉林": 'jilin', '江西': 'jiangxi', '甘肃': 'gansu', '新疆': 'xinjiang',
#      '河北': 'hebei', '海南': 'hainan', '山西': "shanxi", '内蒙古': 'neimenggu', '青海': 'qinghai', '西藏': 'xizang'})
# userProfile_test['age'] = userProfile_test['age'].map(
# {'60后': "60hou", '70后': "70hou", '80后': '80hou', '90后': '90hou', '00后': '00hou'})
# 将性别和省份转换为OneHot编码方式
# userProfile_test = pd.get_dummies(userProfile_test, columns=['age', 'gender'])
action_test_new = open(project_path + 'preprocess/middle_file/' + 'action_test_new.csv', 'w')
action_test_new.write(
    'userid,actionType1,actionType2,actionType3,actionType4,actionType5,actionType6,actionType7,actionType8,actionType9')
for userid in action_test['userid'].unique():
    actiontype_arr = np.array(action_test[action_test['userid'] == userid]['actionType'])
    cnt = Counter()
    for actiontype in actiontype_arr:
        cnt[actiontype] += 1
    action_test_new.write('\n')
    action_test_new.write(
        str(userid) + ',' + str(cnt[1]) + ',' + str(cnt[2]) + ',' + str(cnt[3]) + ',' + str(cnt[4]) + ',' + str(
            cnt[5]) + ',' + str(cnt[6]) + ',' + str(cnt[7]) + ',' + str(cnt[8]) + ',' + str(cnt[9]))
action_test_new.close()
action_test = pd.read_csv(project_path + 'preprocess/middle_file/' + 'action_test_new.csv')
test_data = pd.merge(action_test, userProfile_test, how='left', on="userid")
test_data = pd.merge(test_data, orderFuture_test, how='left', on="userid")
del test_data['age']
del test_data['gender']
del test_data['province']
test_data.to_csv(testdata_output_path + "testdata_base.csv", index=False)
