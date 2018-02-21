#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-18 20:16:39
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import time
import xgboost as xgb

project_path = 'E:\Document\Competition\DC_Mac\DC_Code\\'
train_data = pd.read_csv(project_path + 'preprocess/traindata_output/traindata_output.csv')
test_data = pd.read_csv(project_path + 'preprocess/testdata_output/testdata_output.csv')
print('开始训练...')
params = {
    #         'tree_method':'gpu_hist',
    'learning_rate': 0.01,
    'n_estimators': 150,
    'max_depth': 8,
    'min_child_weight': 3,
    'gamma': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'silent': 1,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'scale_pos_weight': 0.5
}
print('开始CV 10折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_data.shape[0])
test_preds = np.zeros((test_data.shape[0], 10))
kf = KFold(n_splits=10, shuffle=True, random_state=1015)
k = kf.split(train_data)
predictors = [f for f in test_data.columns if f not in ['orderType']]
# print(predictors)
for i, (train_index, test_index) in enumerate(k):
    print('第{}次训练...'.format(i))
    #     print(train_index)
    #     print(test_index)
    train_feat1 = train_data.iloc[train_index]
    train_feat2 = train_data.iloc[test_index]
    xgb_train1 = xgb.DMatrix(train_feat1[predictors], label=train_feat1['orderType'])  # , categorical_feature=['性别'])
    xgb_train2 = xgb.DMatrix(train_feat2[predictors], label=train_feat2['orderType'])  # , categorical_feature=['性别'])
    gbm = xgb.train(params,
                    xgb_train1,
                    num_boost_round=10000,
                    evals=[(xgb_train1, 'train_auc'), (xgb_train2, 'test_auc')],
                    verbose_eval=100,
                    early_stopping_rounds=120,
                    )
    watchlist = [(xgb_train1, 'train_auc'), (xgb_train2, 'test_auc')]
    print('start training...')
    #     gbm = xgb.train(params, xgb_train1, num_boost_round=800, evals=watchlist, early_stopping_rounds=40)
    feat_imp = pd.Series(gbm.get_fscore(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(xgb.DMatrix(train_feat2[predictors]))
    test_preds[:, i] = gbm.predict(xgb.DMatrix(test_data[predictors]))
print('线下得分：    {}'.format(roc_auc_score(train_data['orderType'], train_preds)))
print('CV训练用时{}秒'.format(time.time() - t0))
pred = test_preds.mean(axis=1)
result = pd.DataFrame(columns=['userid', 'orderType'])
result['userid'] = test_data.userid
result['orderType'] = pred
result.to_csv(project_path + 'result/xgb_cv/' + time.strftime("%m-%d-%H-%M", time.localtime()) + '-result' + '.csv',
              index=False)
