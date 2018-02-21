#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-30 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)
# 本题的目的就是利用已知的信息比如：个人信息、行为数据、历史订单信息、评论数据来预测未来是否会选购服务，找到这种映射关系。
# 那么考虑从选购了精品服务orderType1的人群中，去观察他们四个历史数据的特点。


import os
import pandas as pd
import numpy as np
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
