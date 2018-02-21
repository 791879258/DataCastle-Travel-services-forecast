#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-25 16:41:30
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import gc

project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'
train_data = pd.read_csv(project_path + 'preprocess/traindata_output/traindata_output.csv')
# train_data=pd.read_csv(project_path+'fs_traindata.csv')
# train_data.rename(columns={'label':'orderType'},inplace=True)
# feature_selection
# train_data = pd.read_csv(project_path + 'preprocess/traindata_output/traindata_output_variance_sel.csv')

test_data = pd.read_csv(project_path + 'preprocess/testdata_output/testdata_output.csv')
# feature_selection
# test_data = pd.read_csv(project_path + 'preprocess/testdata_output/testdata_output_variance_sel.csv')
print(train_data.shape)
print(test_data.shape)
# 线上提交
train_y = train_data.orderType
del train_data['orderType']
train_X = train_data.iloc[:, 1:]
test_data_xgb = test_data.iloc[:, 1:]

traindata = xgb.DMatrix(train_X, label=train_y)
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'eval_metric': 'auc',
#           'gamma': 0.1,
#           'min_child_weight': 1.1,
#           'max_depth': 5,
#           'subsample': 0.7,
#           'eta': 0.1,
#           'n_estimators': 1000
#           }
params = {
#         'tree_method':'gpu_hist',
        'learning_rate' : 0.01,
        'n_estimators': 150,
        'max_depth': 8,
        'min_child_weight': 3,
        'gamma': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'silent': 1,
        'eval_metric':'auc',
        'objective':'binary:logistic',
        'scale_pos_weight':0.5
        }
gbm = xgb.train(params,
                traindata,
                num_boost_round=3000,
                evals=[(traindata,'train_auc')],
                verbose_eval=100,
                early_stopping_rounds=120,
                )
test_data_matrix = gbm.DMatrix(test_data_xgb)
# watchlist = [(traindata, 'train')]

# print('start training...'
# model = xgb.train(params, traindata, num_boost_round=500, evals=watchlist, early_stopping_rounds=30)
# plt feature importance
# fig, ax = plt.subplots(figsize=(12, 18))
# xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
# plt.show()
# predict test set
# print('start predicting...')
# test_data_matrix = xgb.DMatrix(test_data_xgb)
pred = model.predict(test_data_matrix)

test_data['orderType'] = pred
result = test_data[['userid', 'orderType']]
result.to_csv(project_path + 'result/xgb/' + time.strftime("%m-%d-%H-%M", time.localtime()) + '-result' + '.csv',
              index=False)

# ##########线下验证
# train_y = train_data.orderType
# del train_data['orderType']
# train_X = train_data
#
# X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)
# # X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33)
# X_test_userid = pd.DataFrame(X_test['userid']).reset_index(drop=True)
# X_train = X_train.iloc[:, 1:]
# X_test = X_test.iloc[:, 1:]
#
# traindata = xgb.DMatrix(X_train, label=y_train)
# validdata = xgb.DMatrix(X_test, label=y_test)
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'eval_metric': 'auc',
#           'gamma': 0.1,
#           'min_child_weight': 1.1,
#           'max_depth': 5,
#           'subsample': 0.7,
#           'eta': 0.1,
#           'n_estimators': 1000
#           }
#
# watchlist = [(traindata, 'train')]
#
# print('start training...')
# model = xgb.train(params, traindata, num_boost_round=400, evals=watchlist, early_stopping_rounds=30)
# # feature_score = model.get_fscore()
# # feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
# # feature_score_df = pd.DataFrame.from_dict(feature_score, orient='columns')
# # feature_score_df.rename(columns={0: 'feature', 1: 'score'}, inplace=True)
# # feature_score_df.to_csv(project_path + 'model/xgb_feature_score.csv', index=False)
# # predict test set
# print('start predicting...')
#
# pred = model.predict(validdata)
# result = pd.DataFrame(columns=['userid', 'orderType_true', 'orderType_pred'])
# result['userid'] = X_test_userid['userid']
# result['orderType_true'] = y_test.values
# result['orderType_pred'] = pred
# valid_auc = metrics.roc_auc_score(result['orderType_true'], result['orderType_pred'])  # 验证集上的auc值
# print('valid auc: ' + str(valid_auc))
# # # result.to_csv(project_path + 'result/xgb_valid/' + time.strftime("%m-%d-%H-%M", time.localtime()) + '-result' + '.csv',
# # #               index=False)
