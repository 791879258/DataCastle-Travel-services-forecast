#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-17 10:16:20
# @Author  : guanglinzhou (xdzgl812@163.com)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

project_path = '/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/'

result_rf = pd.read_csv(project_path + 'result/rf/01-17-11-29-result.csv')
result_xgb = pd.read_csv(project_path + 'result/xgb/02-05-16-21-result.csv')

merge_result = pd.merge(result_xgb, result_rf, how='left', on='userid', suffixes=['_xgb', '_rf'])
merge_result['orderType'] = (0.8 * merge_result['orderType_xgb'] + 0.2 * merge_result['orderType_rf'])
del merge_result['orderType_xgb']
del merge_result['orderType_rf']
merge_result.to_csv(project_path + 'result/merge/xgb_rf_' + time.strftime("%m-%d-%H-%M", time.localtime()) + '.csv',
                    index=False)
