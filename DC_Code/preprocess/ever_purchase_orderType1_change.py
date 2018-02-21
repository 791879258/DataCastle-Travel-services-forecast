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
# for userid in traindata_output['userid'].values:
#     if(traindata_output[traindata_output['userid']==userid])