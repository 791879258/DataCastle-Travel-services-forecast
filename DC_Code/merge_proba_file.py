#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-18 20:16:39
# @Author  : guanglinzhou (xdzgl812@163.com)

import os
import pandas as pd
import numpy as np

f_other = pd.read_csv('E:\Document\Competition\DC_Mac\DC_Code\\result\other\96609.csv')
f_me = pd.read_csv('E:\Document\Competition\DC_Mac\DC_Code\\result\\xgb_cv\\02-05-16-21-result.csv')

# bili = [0.6, 0.4]
other_proba = f_other['orderType'].values
me_proba = f_me['orderType'].values
df = pd.DataFrame()
df['userid'] = f_me['userid'].values
df['orderType'] = 0.4 * me_proba + 0.6 * other_proba
df.to_csv('E:\Document\Competition\DC_Mac\DC_Code\\result\\xgb_cv\\merge_46.csv', index=False)
