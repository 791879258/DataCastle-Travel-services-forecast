# coding: utf-8
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import gc
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import preprocessing
import pywt
##read_data###
action_train=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/train/action_train.csv')#用户行为数据
#行为类型一共有9个，其中1是唤醒app；2~4是浏览产品，无先后关系；5~9则是有先后关系的，从填写表单到提交订单再到最后支付。
orderFuture_train=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/train/orderFuture_train.csv')#待预测数据
orderHistory_train=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/train/orderHistory_train.csv', encoding='utf-8')#用户历史订单数据
userComment_train=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/train/userComment_train.csv', encoding='utf-8')#用户评论数据
userProfile_train=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/train/userProfile_train.csv', encoding='utf-8')#用户个人信息

action_test=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/test/action_test.csv')
orderFuture_test=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/test/orderFuture_test.csv')
orderHistory_test=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/test/orderHistory_test.csv', encoding='utf-8')
userComment_test=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/test/userComment_test.csv', encoding='utf-8')
userProfile_test=pd.read_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/data/test/userProfile_test.csv', encoding='utf-8')


def time_conv(x):
    timeArray=time.localtime(x)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

#action_train.actionTime=action_train.actionTime.map(lambda x: time_conv(x))
# orderHistory_train.orderTime=pd.to_datetime(orderHistory_train.orderTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
# orderHistory_test.orderTime=pd.to_datetime(orderHistory_test.orderTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
# action_train.actionTime=pd.to_datetime(action_train.actionTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
# action_test.actionTime=pd.to_datetime(action_test.actionTime.map(lambda x: time_conv(x)),format="%Y-%m-%d %H:%M:%S")
orderFuture_train.rename(columns={'orderType':'label'},inplace=True)

####feature#####
##user过去是否订购过orderType 订购1 未订购0
def orderHistory_feat(df):
    grouped=df[['userid','orderType']].groupby('userid',as_index=False)
    df_count=grouped.count()
    df_count.rename(columns={'orderType':'df_count'},inplace=True)
    df_sum=grouped.sum()
    df_sum.rename(columns={'orderType':'df_sum'},inplace=True)
    df_merge=pd.merge(df_count,df_sum,on='userid',how='left')
    df_merge['rate']=df_merge['df_sum']/df_merge['df_count']
    del df_merge['df_count']
    df_merge.rename(columns={'df_sum':'orderHistory_feat_sum','rate':'orderHistory_feat_rate'},inplace=True)
    gc.collect()
    return df_merge

#每个action的比率 #第一个、最后一个、倒数第二个、倒数第三个action以及相对应的时间
def actions_orderType(df):
    s=df
    s['count']=1
    df_count = s[['userid','count']].groupby('userid',as_index=False).count()
    actionType = pd.get_dummies(s['actionType'],prefix='actionType')
    s = pd.concat([s['userid'],actionType],axis=1)
    s = s.groupby('userid',as_index=False).sum()
    for column in range(1,s.shape[1]):
        s['actionType_{}'.format(column)] = s['actionType_{}'.format(column)]/df_count['count']
    del df['count']
    df_order = df.sort_values(by=['userid', 'actionTime'], axis=0, ascending=True)#桉用户时间排序
    #按时间排名 升序
    df_order['group_sort'] = df_order['actionTime'].groupby(df_order['userid']).rank(ascending=0, method='dense')

    def time_fomat(da,i):
        da['last_'+i+'_Time_fomat'] = pd.to_datetime(da['action_last_'+str(i)+'_Time'].map(lambda x: time_conv(x)),
                                                 format="%Y-%m-%d %H:%M:%S")
        da['action_last_'+str(i)+'_weekofyear'] = da['last_'+str(i)+'_Time_fomat'].map(lambda x: x.weekofyear)
        da['action_last_'+str(i)+'_dayofweek'] = da['last_'+str(i)+'_Time_fomat'].map(lambda x: x.dayofweek)
        da['action_last_'+str(i)+'_hour'] = da['last_'+str(i)+'_Time_fomat'].map(lambda x: x.hour)
        da['action_last_'+str(i)+'_day'] = da['last_'+str(i)+'_Time_fomat'].map(lambda x: x.day)
        da['action_last_'+str(i)+'_month'] = da['last_'+str(i)+'_Time_fomat'].map(lambda x: x.month)
        da['action_last_'+str(i)+'_isweekend'] = da['action_last_'+str(i)+'_dayofweek'].map(lambda x: 1 if ((x == 5) | (x == 6)) else 0)
        del da['last_'+str(i)+'_Time_fomat']
        return da
    # 倒数第一个actiontype和时间
    da = df_order[df_order['group_sort'] == 1]
    del da['group_sort']
    da.columns = ['userid', 'action_last_1', 'action_last_1_Time']
    # 新加
    # da = time_fomat(da,'1')
    s = pd.merge(s, da, how='left', on=['userid'])

    # 倒数第二个actiontype和时间
    da = df_order[df_order['group_sort'] == 2]
    del da['group_sort']
    da.columns = ['userid', 'action_last_2', 'action_last_2_Time']
    # da = time_fomat(da, '2')
    s = pd.merge(s, da, how='left', on=['userid'])

    # 倒数第三个actiontype和时间
    da = df_order[df_order['group_sort'] == 3]
    del da['group_sort']
    da.columns = ['userid', 'action_last_3', 'action_last_3_Time']
    # da = time_fomat(da, '3')
    s = pd.merge(s, da, how='left', on=['userid'])

    #降序
    df_order['group_sort'] = df_order['actionTime'].groupby(df_order['userid']).rank(ascending=1, method='dense')
    # 第一个actiontype和时间
    da = df_order[df_order['group_sort'] == 1]
    del da['group_sort']
    da.columns = ['userid', 'action_first_1', 'action_first_1_Time']
    # da['first_1_Time_fomat'] = pd.to_datetime(da['action_first_1_Time'].map(lambda x: time_conv(x)),
    #                                                  format="%Y-%m-%d %H:%M:%S")
    # da['action_first_1_weekofyear'] = da['first_1_Time_fomat'].map(lambda x: x.weekofyear)
    # da['action_first_1_dayofweek'] = da['first_1_Time_fomat'].map(lambda x: x.dayofweek)
    # da['action_first_1_hour'] = da['first_1_Time_fomat'].map(lambda x: x.hour)
    # da['action_first_1_day'] = da['first_1_Time_fomat'].map(lambda x: x.day)
    # da['action_first_1_month'] = da['first_1_Time_fomat'].map(lambda x: x.month)
    # da['action_first_1_isweekend'] = da['action_first_1_dayofweek'].map(
    #                                   lambda x: 1 if ((x == 5) | (x == 6)) else 0)
    # del da['first_1_Time_fomat']
    s = pd.merge(s, da, how='left', on=['userid'])

    # del s['action_first_1'],s['actionType_9'],s['actionType_3'],s['actionType_8']        #新改的
    gc.collect()
    return s

#对action_Type作小波变换
def WT_actionType(df):
    df = df.sort_values(by=['userid', 'actionTime'], ascending=[1, 0]).reset_index()  # 用户升序，动作时间降序
    del df['index']
    wt={}
    for name, group in df.groupby('userid'):
        group = group.reset_index()
        del group['index']
        dd = list(group['actionType'])
        w = pywt.Wavelet('sym3')
        cA, cD = pywt.dwt(dd, wavelet=w, mode='cpd')
        wt[name] = {}
        wt[name]['wt_1_mean'] = np.mean(cA)
        wt[name]['wt_1_var'] = np.var(cA)
        wt[name]['wt_2_mean'] = np.mean(cD)
        wt[name]['wt_2_var'] = np.var(cD)
    df_wt = pd.DataFrame.from_dict(wt, orient='index').reset_index()
    df_wt.columns = ['userid', 'wt_1_mean','wt_1_var','wt_2_mean','wt_2_var']
    return df_wt

#用户的年龄、性别、省份、 成绩下降一点点
def user_information(df):
    user = df
    # lsex = preprocessing.LabelEncoder()
    # lsex.fit(user['gender'])
    user['gender'] = user['gender'].map({'男': 1, '女': 0})
    user['age'] = user['age'].map({'00后': 0, '90后': 1,'80后': 2, '70后': 3,'60后': 4})
    user['province'] = user['province'].map({'上海':0, '北京':1, '河南':2, '广东':3, '辽宁':4, '陕西':5,
                                             '浙江':6, '四川':7, '云南':8, '黑龙江':9,'安徽':10, '江苏':11,
                                             '重庆':12, '天津':13, '福建':14, '山东':15, '湖北':16, '贵州':17,
                                             '湖南':18, '宁夏':19, '广西':20,'吉林':21, '江西':22, '甘肃':23,
                                             '新疆':24, '河北':25, '海南':26, '山西':27, '内蒙古':28, '青海':29, '西藏':30})
    return user

#时间间隔的均值、方差、最小值、众数、第一个时间间隔、时间间隔倒数第1个值、时间间隔倒数第二个值、时间间隔倒数第三个值、时间间隔倒数第四个值
#最后三个时间间隔均值、最后三个时间间隔方差
def time_jiange(df):
    #存在没有点击时间间隔的 设置为nan或很大的数
    df = df.sort_values(by=['userid', 'actionTime'], axis=0, ascending=True)  # 桉用户时间排序
    df_second = df
    df_second['group_sort'] = df_second['actionTime'].groupby(df_second['userid']).rank(ascending=0, method='dense')#排名
    #去掉最后一行数据
    df_second = df_second[df_second['group_sort'] != 1.0]
    del df_second['group_sort']
    #增加第一行数据
    ss = df.drop_duplicates(['userid']) #(39047) 存在只有一条数据记录
    ss['actionType'] = 0
    ss['actionTime'] = 0
    del ss['group_sort']
    ss.columns = ['userid', 'actionType_last', 'actionTime_last']
    df_second.columns = ['userid', 'actionType_last', 'actionTime_last']
    df_second = pd.concat([df_second,ss])
    df_second = df_second.sort_values(by=['userid', 'actionTime_last'], axis=0, ascending=True)  # 桉用户时间排序

    df['group_sort'] = df['actionTime'].groupby(df['userid']).rank(ascending=1, method='dense')#排名
    del df_second['userid']
    df_second = df_second.reset_index()
    del df_second['index']
    df=pd.concat([df,df_second],axis=1)
    df['time_jiange'] = df['actionTime']-df['actionTime_last']
    df['action_type_jiange'] = df['actionType']-df['actionType_last']
    #去掉第一行 得到时间间隔
    df = df[df['group_sort'] != 1.0]#1294549
    df['group_sort'] = df['actionTime'].groupby(df['userid']).rank(ascending=1, method='dense')#排名

    #时间间隔第一个值
    jiange = pd.concat([df['userid'],df['group_sort'],df['time_jiange'],df['action_type_jiange']],axis=1)#动作时间从小到大
    user = ss['userid'].reset_index()
    del user['index']
    s1=jiange[jiange['group_sort']==1].reset_index()
    del s1['index'],s1['group_sort'],s1['action_type_jiange']
    s1.columns=['userid','time_jiange_f1']
    user = pd.merge(user,s1,how='left',on=['userid'])
    #时间间隔最后一个值
    df['group_sort'] = df['actionTime'].groupby(df['userid']).rank(ascending=0, method='dense')#排名 降序
    jiange = pd.concat([df['userid'],df['group_sort'],df['time_jiange'],df['action_type_jiange']],axis=1)#动作时间从大到小
    s1=jiange[jiange['group_sort']==1].reset_index()
    del s1['index'],s1['group_sort'],s1['action_type_jiange']
    s1.columns=['userid','time_jiange_l1']
    user = pd.merge(user,s1,how='left',on=['userid'])
    #时间间隔倒数第二个值
    s1=jiange[jiange['group_sort']==2].reset_index()
    del s1['index'],s1['group_sort'],s1['action_type_jiange']
    s1.columns=['userid','time_jiange_l2']
    user = pd.merge(user,s1,how='left',on=['userid'])
    #时间间隔倒数第三个值
    s1=jiange[jiange['group_sort']==3].reset_index()
    del s1['index'],s1['group_sort'],s1['action_type_jiange']
    s1.columns=['userid','time_jiange_l3']
    user = pd.merge(user,s1,how='left',on=['userid'])
    #时间间隔倒数第四个值
    s1=jiange[jiange['group_sort']==4].reset_index()
    del s1['index'],s1['group_sort'],s1['action_type_jiange']
    s1.columns=['userid','time_jiange_l4']
    user = pd.merge(user,s1,how='left',on=['userid'])

    #分组统计
    jiange = pd.concat([df['userid'],df['group_sort'],df['time_jiange'],df['action_type_jiange']],axis=1)
    del jiange['group_sort']
    grouped = jiange.groupby('userid')
    t = grouped['time_jiange'].agg(['mean','min','var','median']).reset_index()
    user = pd.merge(user,t,how='left',on=['userid'])
    #求最后三个时间间隔均值、方差
    l3_mean=[]
    l3_var=[]
    for ix,row in user.iterrows():
        data=[]
        if ~np.isnan(row['time_jiange_l1']):
            data.append(row['time_jiange_l1'])
        if ~np.isnan(row['time_jiange_l2']):
            data.append(row['time_jiange_l2'])
        if ~np.isnan(row['time_jiange_l3']):
            data.append(row['time_jiange_l3'])
        if ~np.isnan(row['time_jiange_l4']):
            data.append(row['time_jiange_l4'])
        mean=np.mean(data)
        l3_mean.append(mean)
        var = np.var(data)
        l3_var.append(var)
    l3_mean = pd.DataFrame(l3_mean)
    l3_mean.columns=['l3_mean']
    l3_var = pd.DataFrame(l3_var)
    l3_var.columns=['l3_var']
    user = pd.concat([user,l3_mean],axis=1)
    user = pd.concat([user,l3_var],axis=1)
    gc.collect()
    return user

#离最近的1~9的距离、时间
def jin1_9(df):
    df = df.sort_values(by=['userid', 'actionTime'], ascending=[1, 0]).reset_index() #用户升序，动作时间降序
    del df['index']
    def findi_juli_time(data,i):#查找actionType在data中的最近位置和对应的时间
        ll={}
        ll_time={}
        for name,group in data.groupby('userid'):
            group = group.reset_index()
            del group['index']
            dd = list(group['actionType'])
            if (i in dd):
                ll[name] = dd.index(i)
                ll_time[name] = group.ix[dd.index(i),['actionTime']]
        df_ll = pd.DataFrame.from_dict(ll,orient='index').reset_index()
        df_ll.columns=['userid','juli_'+str(i)]
        df_ll_time = pd.DataFrame.from_dict(ll_time,orient='index').reset_index()
        df_ll_time.columns=['userid','juli_'+str(i)+'_time']
        ddd = pd.merge(df_ll, df_ll_time, how='left', on=['userid'])
        return ddd

    d_1 = findi_juli_time(df, 1)
    d_2 = findi_juli_time(df, 2)
    d_3 = findi_juli_time(df, 3)
    d_4 = findi_juli_time(df, 4)
    d_5 = findi_juli_time(df, 5)
    d_6 = findi_juli_time(df, 6)
    d_7 = findi_juli_time(df, 7)
    d_8 = findi_juli_time(df, 8)
    d_9 = findi_juli_time(df, 9)
    ds = pd.merge(d_1, d_2,how='left',on=['userid'])
    ds = pd.merge(ds, d_3, how='left', on=['userid'])
    ds = pd.merge(ds, d_4, how='left', on=['userid'])
    ds = pd.merge(ds, d_5, how='left', on=['userid'])
    ds = pd.merge(ds, d_6, how='left', on=['userid'])
    ds = pd.merge(ds, d_7, how='left', on=['userid'])
    ds = pd.merge(ds, d_8, how='left', on=['userid'])
    ds = pd.merge(ds, d_9, how='left', on=['userid'])
    # ds = ds.drop(['juli_9','juli_7','juli_3','juli_4','juli_8','juli_6',
    #               'juli_9_time','juli_8_time','juli_3_time','juli_4_time'],axis=1)
    gc.collect()
    return ds

#离最近的1~9的时间间隔的均值、方差、小值、最大值
def juli_time_jiange(df):
    # 存在没有点击时间间隔的 设置为nan或很大的数
    df = df.sort_values(by=['userid', 'actionTime'], axis=0, ascending=True)  # 桉用户时间排序
    df_second = df
    df_second['group_sort'] = df_second['actionTime'].groupby(df_second['userid']).rank(ascending=0,
                                                                                        method='dense')  # 排名
    # 去掉最后一行数据
    df_second = df_second[df_second['group_sort'] != 1.0]
    del df_second['group_sort']
    # 增加第一行数据
    ss = df.drop_duplicates(['userid'])  # (39047) 存在只有一条数据记录
    ss['actionType'] = 0
    ss['actionTime'] = 0
    del ss['group_sort']
    ss.columns = ['userid', 'actionType_last', 'actionTime_last']
    df_second.columns = ['userid', 'actionType_last', 'actionTime_last']
    df_second = pd.concat([df_second, ss])
    df_second = df_second.sort_values(by=['userid', 'actionTime_last'], axis=0, ascending=True)  # 桉用户时间排序

    df['group_sort'] = df['actionTime'].groupby(df['userid']).rank(ascending=1, method='dense')  # 排名
    del df_second['userid']
    df_second = df_second.reset_index()
    del df_second['index']
    df = pd.concat([df, df_second], axis=1)
    df['time_jiange'] = df['actionTime'] - df['actionTime_last']
    df['action_type_jiange'] = df['actionType'] - df['actionType_last']
    # 去掉第一行 得到时间间隔
    df = df[df['group_sort'] != 1.0]  # 1294549
    df = df.sort_values(by=['userid', 'actionTime'], ascending=[1, 0]).reset_index()  # 用户升序，动作时间降序
    # 时间间隔第一个值
    jiange = pd.concat([df['userid'], df['actionType'], df['actionTime'], df['time_jiange']],axis=1)  # 动作时间从小到大

    def find_timejiange(data,i):#查找actionType在data中的最近位置和对应的时间
        ddd=pd.DataFrame()
        for name,group in data.groupby('userid'):
            group = group.reset_index()
            del group['index']
            dd = list(group['actionType'])
            if (i in dd):
                ddd=pd.concat([ddd,group[0:(dd.index(i)+1)]])
        return ddd
    gc.collect()
    user = ss['userid'].reset_index()
    del user['index']
    # 离最近的2的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 2)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min', 'max','var', 'median']).reset_index()#
    t.columns = ['userid','juli_2_mean', 'juli_2_min', 'juli_2_max','juli_2_var', 'juli_2_median']#
    user = pd.merge(user, t, how='left', on=['userid'])
    gc.collect()
    # 离最近的3的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 3)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min', 'max', 'var', 'median']).reset_index()
    t.columns = ['userid', 'juli_3_mean', 'juli_3_min', 'juli_3_max', 'juli_3_var', 'juli_3_median']
    user = pd.merge(user, t, how='left', on=['userid'])
    gc.collect()
    # 离最近的4的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 4)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min', 'max', 'median','var']).reset_index()#
    t.columns = ['userid','juli_4_mean', 'juli_4_min', 'juli_4_max', 'juli_4_median','juli_4_var']#
    user = pd.merge(user, t, how='left', on=['userid'])
    gc.collect()
    #离最近的5的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 5)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min','max', 'var', 'median']).reset_index()
    t.columns=['userid','juli_5_mean','juli_5_min','juli_5_max','juli_5_var','juli_5_median']
    user = pd.merge(user,t,how='left',on=['userid'])
    gc.collect()
    #离最近的6的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 6)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min','max', 'var', 'median']).reset_index()
    t.columns=['userid','juli_6_mean','juli_6_min','juli_6_max','juli_6_var','juli_6_median']
    user = pd.merge(user, t, how='left', on=['userid'])
    gc.collect()
    # 离最近的7的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 7)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min', 'max','var', 'median']).reset_index()#
    t.columns = ['userid', 'juli_7_mean', 'juli_7_min', 'juli_7_max', 'juli_7_var','juli_7_median']#,
    user = pd.merge(user, t, how='left', on=['userid'])
    gc.collect()
    # 离最近的8的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 8)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min','max','var', 'median']).reset_index()#
    t.columns = ['userid','juli_8_mean', 'juli_8_min','juli_8_max','juli_8_var','juli_8_median']#
    user = pd.merge(user, t, how='left', on=['userid'])
    gc.collect()
    # 离最近的9的时间间隔均值、最小、最大、方差、中位数
    d5 = find_timejiange(jiange, 9)
    grouped = d5.groupby('userid')
    t = grouped['time_jiange'].agg(['mean', 'min', 'max','var', 'median']).reset_index()#
    t.columns = ['userid','juli_9_mean', 'juli_9_min', 'juli_9_max','juli_9_var','juli_9_median']#
    user = pd.merge(user, t, how='left', on=['userid'])
    # user['juli_9_mean*var'] = user['juli_9_mean']*user['juli_9_var'] 去掉
    gc.collect()

    return user

#对订单的评论分数，以及是否为全好评(5分)
def comment(df):
    df['GoodorBad'] = df['rating'].apply(lambda x:1 if x==5 else 0)
    com = pd.concat([df['userid'],df['rating'],df['GoodorBad']],axis=1)
    return com

def gen_train_feat():
    actions=orderFuture_train
    actions=pd.merge(actions,orderHistory_feat(orderHistory_train),on='userid',how='left')
    actions=pd.merge(actions,actions_orderType(action_train),on='userid',how='left')
    ###add feature###
    #年龄 省份 性别
    # actions = pd.merge(actions, user_information(userProfile_train), on='userid', how='left')
    actions = pd.merge(actions, time_jiange(action_train), on='userid', how='left')
    actions = pd.merge(actions, jin1_9(action_train), on='userid', how='left')
    actions = pd.merge(actions, juli_time_jiange(action_train), on='userid', how='left')
    #小波变换
    # actions = pd.merge(actions, WT_actionType(action_train), on='userid', how='left')
    #订单的评论分数，以及是否为全好评(5分)
    actions = pd.merge(actions, comment(userComment_train), on='userid', how='left')
    return actions

def gen_test_feat():
    actions=orderFuture_test
    actions=pd.merge(actions,orderHistory_feat(orderHistory_test),on='userid',how='left')
    actions=pd.merge(actions,actions_orderType(action_test),on='userid',how='left')
    # 年龄 省份 性别
    # actions = pd.merge(actions, user_information(userProfile_test), on='userid', how='left')
    actions = pd.merge(actions, time_jiange(action_test), on='userid', how='left')
    actions = pd.merge(actions, jin1_9(action_test), on='userid', how='left')
    actions = pd.merge(actions, juli_time_jiange(action_test), on='userid', how='left')
    #小波变换
    # actions = pd.merge(actions, WT_actionType(action_test), on='userid', how='left')
    # 订单的评论分数，以及是否为全好评(5分)
    actions = pd.merge(actions, comment(userComment_test), on='userid', how='left')
    return actions


# train_data=gen_train_feat()
# train_data.to_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/fs_traindata.csv',index=False)
test_data=gen_test_feat()
test_data.to_csv('/Users/guanglinzhou/Desktop/DC_Mac/DC_Code/fs_testdata.csv',index=False)
# test_data=gen_test_feat()
# train_label=train_data['label']
# del train_data['label']
# print(train_data.shape)
# print(train_data.columns)
# print(test_data.shape)
# print(test_data.columns)
# gc.collect()


# x_train,x_val,y_train,y_val=train_test_split(train_data,train_label,test_size=0.2,random_state=1015)

# print ('start running ....')
# dtrain = xgb.DMatrix(x_train,label=y_train)
# dval = xgb.DMatrix(x_val,label=y_val)

# # dtrain = xgb.DMatrix(train_data,label=train_label)
# param = {'learning_rate' : 0.01,
#         'n_estimators': 100,
#         'max_depth': 8,
#         'min_child_weight': 5,
#         'gamma': 1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'silent': 1,
#         'objective':'binary:logistic'}

# watchList = [ (dtrain, 'train'), (dval, 'eval')]
# # watchList = [ (dtrain, 'train')]
# plst = list(param.items()) + [('eval_metric', 'auc')]
# bst = xgb.train(plst, dtrain, 10000, watchList, early_stopping_rounds=100)
# gc.collect()
# print('save model......')
# joblib.dump(bst, 'E:\\game\\huangbaoche\\model\\xgb_model_1_15_1.m')
# # bst = joblib.load("E:\\game\\huangbaoche\\model\\xgb_model_1_11_2.m")
# #预测结果
# dtest = xgb.DMatrix(test_data)
# y = bst.predict(dtest)

# orderFuture_test['orderType']=y
# orderFuture_test.to_csv(r'E:\\game\\huangbaoche\\answer\\base{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                   index=False, float_format='%.4f')

# #feature importance
# feat_imp = pd.Series(bst.get_fscore()).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importance')
# plt.ylabel('Feature Importance Score')
# plt.show()