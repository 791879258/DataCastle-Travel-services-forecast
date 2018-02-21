#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-29 12:19:53
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import gc
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2

project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'
traindata_output_path = project_path + "preprocess/traindata_output/"
testdata_output_path = project_path + "preprocess/testdata_output/"

# variance_selection
orin_traindata_output = pd.read_csv(traindata_output_path + 'traindata_output.csv')
orin_testdata_output = pd.read_csv(testdata_output_path + 'testdata_output.csv')
traindata_output = pd.read_csv(traindata_output_path + 'traindata_output.csv')
traindata_output.fillna(0, inplace=True)
y = traindata_output['orderType']
userid_series = orin_traindata_output['userid']
del traindata_output['userid']
del traindata_output['orderType']
X = traindata_output
variance_sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
variance_data = variance_sel.fit_transform(X)
columns_mask = variance_sel.get_support()  # true表示特征未被筛选，false表示特征被剔除了

drop_columnname_list = [X.columns[i] for i in range(len(columns_mask)) if
                        (columns_mask[i] == False)]
print('删除特征数：' + str(len(drop_columnname_list)))
print('删除特征为：')
for name in drop_columnname_list:
    del orin_traindata_output[name]
    del orin_testdata_output[name]
    print(name)

orin_traindata_output.to_csv(traindata_output_path + 'traindata_output_variance_sel.csv', index=False)
orin_testdata_output.to_csv(testdata_output_path + 'testdata_output_variance_sel.csv', index=False)


# # chi2_selection
#
# orin_traindata_output = pd.read_csv(traindata_output_path + 'traindata_output.csv')
# traindata_output = pd.read_csv(traindata_output_path + 'traindata_output.csv')
# traindata_output.fillna(0, inplace=True)
# y = traindata_output['orderType']
# del traindata_output['userid']
# del traindata_output['orderType']
# X = traindata_output
#
# select_k_best_classifier = SelectKBest(score_func=chi2, k=70)
# chi2_data = select_k_best_classifier.fit_transform(X, y)
# userid_series = orin_traindata_output['userid']
# orderType_series = orin_traindata_output['orderType']
# del orin_traindata_output['userid']
# del orin_traindata_output['orderType']
# columns_mask = select_k_best_classifier.get_support()
# drop_columnname_list = [X.columns[i] for i in range(len(columns_mask)) if
#                         (columns_mask[i] == False)]
#
# print('删除特征数：' + str(len(drop_columnname_list)))
# print('删除特征为：')
# for name in drop_columnname_list:
#     del orin_traindata_output[name]
#     print(name)
#
# orin_traindata_output = pd.concat([userid_series, orin_traindata_output, orderType_series], axis=1)
# orin_traindata_output.to_csv(traindata_output_path + 'traindata_output_chi2_sel.csv', index=False)
