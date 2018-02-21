#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-15 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn import preprocessing
import datetime

project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'
traindata_path = project_path + "data/train/"
traindata_path = project_path + "data/train/"
testdata_path = project_path + "data/test/"
feature_file_path = project_path + 'preprocess/feature_file/'

action_train = pd.read_csv(traindata_path + 'action_train.csv')
action_test = pd.read_csv(testdata_path + 'action_test.csv')

orderHistory_train = pd.read_csv(traindata_path + 'orderHistory_train.csv')
orderHistory_test = pd.read_csv(testdata_path + 'orderHistory_test.csv')


def func(df):
    df.sort_values(by=['actionTime'], ascending=True, inplace=True)
    return df


action_train = action_train.groupby('userid').apply(lambda X: func(X))
action_test = action_test.groupby('userid').apply(lambda X: func(X))
action_train.reset_index(drop=True, inplace=True)
action_test.reset_index(drop=True, inplace=True)


def func(df):
    df.sort_values(by=['orderTime'], ascending=True, inplace=True)
    return df


orderHistory_train = orderHistory_train.groupby('userid').apply(
    lambda X: func(X))

orderHistory_test = orderHistory_test.groupby('userid').apply(
    lambda X: func(X))

orderHistory_train.reset_index(drop=True, inplace=True)
orderHistory_test.reset_index(drop=True, inplace=True)

action_train.to_csv(traindata_path + 'action_train.csv', index=False)
action_test.to_csv(testdata_path + 'action_test.csv', index=False)
orderHistory_train.to_csv(traindata_path + 'orderHistory_train.csv', index=False)
orderHistory_test.to_csv(testdata_path + 'orderHistory_test.csv', index=False)
