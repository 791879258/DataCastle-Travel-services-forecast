# encoding=utf-8

import os
import datetime
import pandas as pd
from data_process import *

current_time = "2017-09-12 00:00:00"
early_time = "2016-08-13 00:00:00"

time_format = "%Y-%m-%d %H:%M:%S"

feature_set = ["gender", "age", "province", 'total_order_count', 'common_order_count',
    'special_order_count', 'has_ordered_special', 
    'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6', 'action_7', 'action_8', 'action_9', 
    "action_1_ratio", "action_2_ratio", "action_3_ratio",
    "action_4_ratio", "action_5_ratio", "action_6_ratio", "action_7_ratio", "action_8_ratio", 
    "action_9_ratio", "total_action_count", 
#    "last_special_time_distance", 
    "first_action_type",
    "last_action_type",
    "last_but_one_action_type", 
    "special_order_time_distance_mean", 
    "action_time_distance_mean", "action_time_distance_std", "action_time_distance_min", 
#    "action_1_time_distance_mean", "action_1_time_distance_std", "action_1_time_distance_min", 
#    "action_2_time_distance_mean", "action_2_time_distance_std", "action_2_time_distance_min", 
#    "action_3_time_distance_mean", "action_3_time_distance_std", "action_3_time_distance_min", 
#    "action_4_time_distance_mean", "action_4_time_distance_std", "action_4_time_distance_min", 
#    "action_5_time_distance_mean", "action_5_time_distance_std", "action_5_time_distance_min", 
#    "action_6_time_distance_mean", "action_6_time_distance_std", "action_6_time_distance_min", 
#    "action_7_time_distance_mean", "action_7_time_distance_std", "action_7_time_distance_min", 
#    "action_8_time_distance_mean", "action_8_time_distance_std", "action_8_time_distance_min", 
#    "action_9_time_distance_mean", "action_9_time_distance_std", "action_9_time_distance_min", 
    "action_5_time_distance_mean",
    "action_6_time_distance_mean",
    "last_but_two_action_type", "last_but_three_action_type", 
    "last_order_type", 
    'action_from_last_order_1', 
    'action_from_last_order_2', 'action_from_last_order_3', 'action_from_last_order_4', 
    'action_from_last_order_5', 'action_from_last_order_6', 'action_from_last_order_7', 
    'action_from_last_order_8', 'action_from_last_order_9', 
    "last_two_actions_time_distance",
    "last_but_two_actions_time_distance", 
    "last_but_three_actions_time_distance", 
    "last_but_four_actions_time_distance", "last_four_time_distance_mean", "last_four_time_distance_std",
    "time_distance_from_last_actiontype_1", "time_distance_from_last_actiontype_2",
    "time_distance_from_last_actiontype_3", "time_distance_from_last_actiontype_4",
    "time_distance_from_last_actiontype_5",
    "time_distance_from_last_actiontype_6",
    "time_distance_from_last_actiontype_7", 
    "time_distance_from_last_actiontype_8",
    "time_distance_from_last_actiontype_9",
    "special_time_%s_action_1" % 1440, "special_time_%s_action_2" % 1440, 
    "special_time_%s_action_3" % 1440, "special_time_%s_action_4" % 1440, 
    "special_time_%s_action_5" % 1440, "special_time_%s_action_6" % 1440, 
    "special_time_%s_action_7" % 1440, "special_time_%s_action_8" % 1440, 
    "special_time_%s_action_9" % 1440,
    "special_time_%s_action_1" % 9080, "special_time_%s_action_2" % 9080, 
    "special_time_%s_action_3" % 9080, "special_time_%s_action_4" % 9080, 
    "special_time_%s_action_5" % 9080, "special_time_%s_action_6" % 9080, 
    "special_time_%s_action_7" % 9080, "special_time_%s_action_8" % 9080, 
    "special_time_%s_action_9" % 9080,
    "time_distance_from_first_actiontype", 
    #"special_order_rate",
    #"last_four_action_type_string",
    #"comment_num", "tag_num",
    #"city_diversity", "country_diversity", "continent_diversity",
    "continent_0", "continent_1", "continent_2", "continent_3", "continent_4", "continent_5",
    "last_continuous_count_600",
    "action_1_ratio_from_last_order", 
    "action_2_ratio_from_last_order", 
    "action_3_ratio_from_last_order",
    "action_4_ratio_from_last_order", 
    "action_5_ratio_from_last_order", 
    "action_6_ratio_from_last_order",
    "action_7_ratio_from_last_order", 
    "action_8_ratio_from_last_order", 
    "action_9_ratio_from_last_order",
    "total_action_count_from_last_order", 
    "hour",
    "last_order_time_distance",
#    "day",
#    "comment_time_distance", 
#    "rating", "has_comment", "has_tag", 
    ]
#feature_set.extend(["country_%d" % i for i in range(0,51)])

def get_hour(x):
    return int(x.split(" ")[1].split(":")[0])

def get_day(x):
    return x.split(" ")[0].split("-")[2]

def get_last_time_hour(tag="train") :
    """
    features           comment
    hour       最后一个action的小时时刻 
    """
    path = "cache/%s_last_time_hour.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        actions = pd.read_csv("data/trainingset/action_train.csv")
        futures = pd.read_csv("data/trainingset/orderFuture_train.csv")
        features = pd.merge(actions, futures, on=["userid"], how="left")
        features.sort_values(by=["actionTime"], inplace=True)
        features.drop_duplicates(["userid"], keep="last", inplace=True)
        features["hour"] = features["actionTime"].map(get_hour)
        features = features[["userid", "hour"]]
        features.to_csv(path, index=False)
    return features

def get_last_time_day(tag="train") :
    """
    features           comment
    day       最后一个action的天
    """
    path = "cache/%s_last_time_day.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        actions = pd.read_csv("data/trainingset/action_train.csv")
        features = actions.sort_values(by=["actionTime"])
        features.drop_duplicates(["userid"], keep="last", inplace=True)
        features["day"] = features["actionTime"].map(get_day)
        features = features[["userid", "day"]]
        features.to_csv(path, index=False)
    return features

def time_distance(t1, t2) :
    time_delta = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S") -  \
                    datetime.datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
    seconds = time_delta.days * 24 * 60 * 60 + time_delta.seconds
    return seconds

def long_time_distance(x, T) :
    x.sort_values(["actionTime"], ascending=False, inplace=True)
    time_array = list(x["actionTime"].values)
    if len(time_array) > 5 :
        last_time = time_array[0]
        for i in range(1, len(time_array)) : 
            time = time_array[i]
            if time_distance(time, last_time) > T :
                behind = x.iloc[0 : i]
                before = x.iloc[i : ]
                if len(before[before["actionType"] == 6]) == 0 \
                    and len(behind[behind["actionType"] == 6] > 0) : 
                        return True
            last_time = time 
    else : 
        return False


def time_distance_mean(x, time_column) :
    x = x.sort_values(by=time_column)
    if len(x) > 1 :
        time_array = x[time_column].values
        last_time = time_array[0]
        distances = []
        for i in range(1, len(time_array)):
            time = time_array[i]
            distance = time_distance(last_time, time)
            distances.append(distance)
            last_time = time
        return np.mean(distances)
    else:
        return 25920000    


def time_distance_std(x, time_column) :
    x = x.sort_values(by=[time_column])
    if len(x) > 1 :
        time_array = x[time_column].values
        last_time = time_array[0]
        distances = []
        for i in range(1, len(time_array)):
            time = time_array[i]
            distance = time_distance(last_time, time)
            distances.append(distance)
            last_time = time
        return np.std(distances)
    else:
        return 25920000    

def time_distance_min(x, time_column) :
    x = x.sort_values(by=[time_column])
    if len(x) > 1 :
        time_array = x[time_column].values
        last_time = time_array[0]
        distances = []
        for i in range(1, len(time_array)):
            time = time_array[i]
            distance = time_distance(last_time, time)
            distances.append(distance)
            last_time = time
        return np.min(distances)
    else:
        return 25920000    

def time_distance_max(x, time_column) :
    x = x.sort_values(by=[time_column])
    if len(x) > 1 :
        time_array = x[time_column].values
        last_time = time_array[0]
        distances = []
        for i in range(1, len(time_array)):
            time = time_array[i]
            distance = time_distance(last_time, time)
            distances.append(distance)
            last_time = time
        return np.max(distances)
    else:
        return 25920000    

def last_four_action_type_string(x, time_column) :
    x = x.sort_values(by=[time_column], ascending=False)
    if len(x) > 3 :
        time_array = x["actionType"].values
        result = int(time_array[0])
        for i in range(1, 4):
            result = result * 10 + int(time_array[i])
        return result
    else:
        return 0    

def continuous_count(x, T) :
    x = x.sort_values(by=["actionTime"], ascending=False)
    time_array = list(x["actionTime"].values)
    if len(x) > 1 :
        last_time = time_array[0]
        for i in range(1, len(time_array)) : 
            time = time_array[i]        
            if time_distance(time, last_time) > T:
                return i;
            last_time = time
    else:
        return 1
    return len(time_array)

def get_last_action_continuous_count_features(T, tag="train") :
    """
    features                                comment
    last_continuous_count_T           最后在连续行为数（时间间隔在T范围内）
    """
    path = "cache/%s_last_action_continuous_count_%s.csv" % (tag, T)
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        features = pd.DataFrame(data["userid"].unique())
        features.columns = ["userid"]
        features["last_continuous_count_%s" % T] = data.groupby(["userid"],as_index=False).apply(
                lambda x : continuous_count(x, T))
        features.to_csv(path, index=False)
    return features



def get_last_four_action_type_string_features(tag="train") :
    """
    features                                comment
    last_four_action_type_string        最后四个actiontype的字符串组合
    """
    path = "cache/%s_last_four_action_type_string.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        features = pd.DataFrame()
        features["userid"] = data["userid"].unique()
        data = data.groupby(["userid"], as_index=False)
        features["last_four_action_type_string"] = data.apply(lambda x : last_four_action_type_string(x, "actionTime"))
        features.to_csv(path, index=False)
    return features



def get_special_time_action_count_features(time_gap, tag="train") :
    '''
    '''
    path = "cache/%s_special_time_action_count_%s.csv" % (tag, time_gap)
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        last = data.sort_values(by=["actionTime"])
        last.drop_duplicates(["userid"], keep="last", inplace=True)
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "last_time"}, inplace=True)
        last["special_time"] = last["last_time"].map(
                    lambda x : (datetime.datetime.strptime(x, time_format) -  \
                    datetime.timedelta(minutes=time_gap)).strftime(time_format))

        features = pd.merge(last, data, on=["userid"], how="left")
        features = features[features["actionTime"] > features["special_time"]]
        features = features[["userid", "actionType"]]
        df = pd.get_dummies(features['actionType'], prefix='special_time_%s_action' % time_gap)
        features = pd.concat([features, df], axis=1) 
        features = features.groupby(['userid'], as_index=False).sum()
        features = features[["userid", "special_time_%s_action_1" % time_gap,
                            "special_time_%s_action_2" % time_gap, "special_time_%s_action_3" % time_gap, 
                            "special_time_%s_action_4" % time_gap, "special_time_%s_action_5" % time_gap,
                            "special_time_%s_action_6" % time_gap, "special_time_%s_action_7" % time_gap,
                            "special_time_%s_action_8" % time_gap, "special_time_%s_action_9" % time_gap]]
        features.to_csv(path, index=False)
    return features

def get_time_distance_from_last_actionType_features(i, tag="train") :
    """
    features                                     comment
    time_distance_from_last_actiontype_i    最后actionType i距离现在的时间距离
    """
    path = "cache/%s_time_distance_from_last_actionType_%s.csv" % (tag, i)
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        last = data.sort_values(by=["actionTime"])
        last.drop_duplicates(["userid"], keep="last", inplace=True)
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "last_time"}, inplace=True)

        data = data[data["actionType"] == i]
        data.sort_values(by=["actionTime"], inplace=True)
        data.drop_duplicates(["userid"], keep="last", inplace=True)
        data = data[["userid", "actionTime"]]
        features = pd.merge(last, data, on=["userid"], how="left")
        features["actionTime"].fillna(early_time, inplace=True)
        features["time_distance_from_last_actiontype_%s" % i] = features.apply( \
                lambda x : time_distance(x["actionTime"], x["last_time"]), axis=1)
        features = features[["userid", "time_distance_from_last_actiontype_%s" % i]]
        features.to_csv(path, index=False)
    return features

def get_time_distance_from_first_actiontype_features(tag="train") :
    """
    features                                     comment
    time_distance_from_first_actiontype    第一次action距离最后一次的距离，即使用app时间
    """
    path = "cache/%s_time_distance_from_first_actiontype.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        last = data.sort_values(by=["actionTime"])
        last.drop_duplicates(["userid"], keep="last", inplace=True)
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "last_time"}, inplace=True)

        data.sort_values(by=["actionTime"], inplace=True)
        data.drop_duplicates(["userid"], keep="first", inplace=True)
        data = data[["userid", "actionTime"]]
        features = pd.merge(last, data, on=["userid"], how="left")
        features["time_distance_from_first_actiontype"] = features.apply( \
                lambda x : time_distance(x["actionTime"], x["last_time"]), axis=1)
        features = features[["userid", "time_distance_from_first_actiontype"]]
        features.to_csv(path, index=False)
    return features

def get_last_four_time_distance_mean(tag="train") :
    """
    features                                comment
    last_three_time_distance_mean       最后四个action的时间间隔均值
    """
    path = "cache/%s_last_four_time_distance_mean.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        last = pd.read_csv("cache/%s_last_two_action_time_distance.csv" % tag)
        last_but_one = pd.read_csv("cache/%s_last_but_two_action_time_distance.csv" % tag)
        last_but_two = pd.read_csv("cache/%s_last_but_three_action_time_distance.csv" % tag)
        last_but_three = pd.read_csv("cache/%s_last_but_four_action_time_distance.csv" % tag)
        features = pd.merge(last, last_but_one, on=["userid"], how="left")
        features = pd.merge(features, last_but_two, on=["userid"], how="left")
        features = pd.merge(features, last_but_three, on=["userid"], how="left")
        features["last_four_time_distance_mean"] = features[["last_two_actions_time_distance", "last_but_two_actions_time_distance",
                            "last_but_three_actions_time_distance", "last_but_four_actions_time_distance"]].mean(axis=1)
        features = features[["userid", "last_four_time_distance_mean"]]
        features.to_csv(path, index=False)
    return features

def get_last_four_time_distance_std(tag="train") :
    """
    features                                comment
    last_four_time_distance_std       最后四个action的时间间隔方差
    """
    path = "cache/%s_last_four_time_distance_std.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        last = pd.read_csv("cache/%s_last_two_action_time_distance.csv" % tag)
        last_but_one = pd.read_csv("cache/%s_last_but_two_action_time_distance.csv" % tag)
        last_but_two = pd.read_csv("cache/%s_last_but_three_action_time_distance.csv" % tag)
        last_but_three = pd.read_csv("cache/%s_last_but_four_action_time_distance.csv" % tag)
        features = pd.merge(last, last_but_one, on=["userid"], how="left")
        features = pd.merge(features, last_but_two, on=["userid"], how="left")
        features = pd.merge(features, last_but_three, on=["userid"], how="left")
        features["last_four_time_distance_std"] = features[["last_two_actions_time_distance", "last_but_two_actions_time_distance",
                            "last_but_three_actions_time_distance", "last_but_four_actions_time_distance"]].std(axis=1)
        features = features[["userid", "last_four_time_distance_std"]]
        features.to_csv(path, index=False)
    return features

def get_last_two_action_time_distance_features(tag="train") :
    """
    features                                comment
    last_two_actions_time_distance    最后两个action的时间间隔
    """
    path = "cache/%s_last_two_action_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        last = data.drop_duplicates(["userid"], keep="last")
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "time2"}, inplace=True)
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        last_but_one = data.drop_duplicates(["userid"], keep="last")
        last_but_one = last_but_one[["userid", "actionTime"]]
        last_but_one.rename(columns={"actionTime" : "time1"}, inplace=True)
        features = pd.merge(last, last_but_one, on=["userid"], how="left")
        features["time1"].replace(np.nan, early_time, inplace=True)
        #features = features[features["time1"].notnull()]
        features["last_two_actions_time_distance"] = features.apply( \
                    lambda x : time_distance(x["time1"],x["time2"]), axis=1)
        features = features[["userid", "last_two_actions_time_distance"]]
        features.to_csv(path, index=False)
    return features
    
def get_last_but_two_action_time_distance_features(tag="train") :
    """
    features                                comment
    last_but_two_actions_time_distance    倒数第二与倒数第三个action的时间间隔
    """
    path = "cache/%s_last_but_two_action_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        last = data.drop_duplicates(["userid"], keep="last")
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "time2"}, inplace=True)
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        last_but_one = data.drop_duplicates(["userid"], keep="last")
        last_but_one = last_but_one[["userid", "actionTime"]]
        last_but_one.rename(columns={"actionTime" : "time1"}, inplace=True)
        features = pd.merge(last, last_but_one, on=["userid"], how="left")
        features["time1"].replace(np.nan, early_time, inplace=True)
        #features = features[features["time1"].notnull()]
        features["last_but_two_actions_time_distance"] = features.apply( \
                    lambda x : time_distance(x["time1"],x["time2"]), axis=1)
        features = features[["userid", "last_but_two_actions_time_distance"]]
        features.to_csv(path, index=False)
    return features

def get_last_but_three_action_time_distance_features(tag="train") :
    """
    features                                comment
    last_but_three_actions_time_distance    倒数第四与倒数第三个action的时间间隔
    """
    path = "cache/%s_last_but_three_action_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        last = data.drop_duplicates(["userid"], keep="last")
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "time2"}, inplace=True)
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        last_but_one = data.drop_duplicates(["userid"], keep="last")
        last_but_one = last_but_one[["userid", "actionTime"]]
        last_but_one.rename(columns={"actionTime" : "time1"}, inplace=True)
        features = pd.merge(last, last_but_one, on=["userid"], how="left")
        features["time1"].replace(np.nan, early_time, inplace=True)
        #features = features[features["time1"].notnull()]
        features["last_but_three_actions_time_distance"] = features.apply( \
                    lambda x : time_distance(x["time1"],x["time2"]), axis=1)
        features = features[["userid", "last_but_three_actions_time_distance"]]
        features.to_csv(path, index=False)
    return features

def get_last_but_four_action_time_distance_features(tag="train") :
    """
    features                                comment
    last_but_four_actions_time_distance    倒数第四与倒数第五个action的时间间隔
    """
    path = "cache/%s_last_but_four_action_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        last = data.drop_duplicates(["userid"], keep="last")
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "time2"}, inplace=True)
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        last_but_one = data.drop_duplicates(["userid"], keep="last")
        last_but_one = last_but_one[["userid", "actionTime"]]
        last_but_one.rename(columns={"actionTime" : "time1"}, inplace=True)
        features = pd.merge(last, last_but_one, on=["userid"], how="left")
        features["time1"].replace(np.nan, early_time, inplace=True)
        #features = features[features["time1"].notnull()]
        features["last_but_four_actions_time_distance"] = features.apply( \
                    lambda x : time_distance(x["time1"],x["time2"]), axis=1)
        features = features[["userid", "last_but_four_actions_time_distance"]]
        features.to_csv(path, index=False)
    return features

def get_action_time_distance_features(tag="train") :
    """
    features                                comment
    action_time_distance_mean    用户action平均时间间隔
    action_time_distance_std     用户action时间方差
    """
    path = "cache/%s_action_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        features = pd.DataFrame()
        features["userid"] = data["userid"].unique()
        data = data.groupby(["userid"], as_index=False)
        features["action_time_distance_mean"] = data.apply(lambda x : time_distance_mean(x, "actionTime"))
        features["action_time_distance_std"] = data.apply(lambda x : time_distance_std(x, "actionTime"))
        features["action_time_distance_min"] = data.apply(lambda x : time_distance_min(x, "actionTime"))
        features["action_time_distance_max"] = data.apply(lambda x : time_distance_max(x, "actionTime"))
#        features["action_time_distance_skew"] = data.apply(lambda x : time_distance_skew(x, "actionTime"))
#        features["action_time_distance_kurt"] = data.apply(lambda x : time_distance_kurt(x, "actionTime"))
        features.to_csv(path, index=False)
    return features

def get_action_type_time_distance_features(i, tag="train") :
    """
    features                                comment
    action_i_time_distance_mean    用户action_i平均时间间隔
    action_i_time_distance_std     用户action_i时间方差
    """
    path = "cache/%s_action_time_distance_%s.csv" % (tag, i)
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        features = pd.DataFrame()
        features["userid"] = data["userid"].unique()
        data = data[data["actionType"] == i]
        data = data.groupby(["userid"], as_index=False)
        features["action_%s_time_distance_mean" % i] = data.apply(lambda x : time_distance_mean(x, "actionTime"))
        features["action_%s_time_distance_std" % i] = data.apply(lambda x : time_distance_std(x, "actionTime"))
        features["action_%s_time_distance_min" % i] = data.apply(lambda x : time_distance_min(x, "actionTime"))
        features["action_%s_time_distance_max" % i] = data.apply(lambda x : time_distance_max(x, "actionTime"))
#        features["action_time_distance_skew"] = data.apply(lambda x : time_distance_skew(x, "actionTime"))
#        features["action_time_distance_kurt"] = data.apply(lambda x : time_distance_kurt(x, "actionTime"))
        features.to_csv(path, index=False)
    return features

def get_special_order_time_distance_mean_features(tag="train") :
    """
    features                                comment
    special_order_time_distance_mean    购买精美服务的平均时间间隔
    """
    path = "cache/%s_special_order_time_distance_mean.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(history_train_path)
        else :
            data = pd.read_csv(history_test_path)
        data = data[data["orderType"] == 1]
        features = pd.DataFrame()
        features["userid"] = data["userid"].unique()
        data = data.groupby(["userid"], as_index=False)
        features["special_order_time_distance_mean"] = data.apply(lambda x : time_distance_mean(x, "orderTime"))
        features.to_csv(path, index=False)
    return features

def get_last_special_order_time_distance_features(tag="train") :
    """
    features                        comment
    last_special_time_distance      上一次购买精美服务距离当前有多久
    """
    path = "cache/%s_last_special_order_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(history_train_path)
        else :
            data = pd.read_csv(history_test_path)
        data = data[data["orderType"] == 1]
        data = data.sort_values(by=["orderTime"])
        data = data.drop_duplicates(["userid"], keep="last")
        data["last_special_time_distance"] = data["orderTime"].map(
                lambda x : time_distance(x, current_time))
        features = data[["userid", "last_special_time_distance"]]
        features.to_csv(path, index=False)
    return features

def get_last_order_time_distance_features(tag="train") :
    """
    features                        comment
    last_order_time_distance      上一次购买服务距离当前有多久
    """
    path = "cache/%s_last_order_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            actions = pd.read_csv(action_train_path)
            data = pd.read_csv(history_train_path)
        else :
            actions = pd.read_csv(action_test_path)
            data = pd.read_csv(history_test_path)
        actions.sort_values(by=["actionTime"], inplace=True)
        actions.drop_duplicates(["userid"], inplace=True)
        actions = actions[["userid", "actionTime"]]
        data = data.sort_values(by=["orderTime"])
        data = data.drop_duplicates(["userid"], keep="last")
        data = pd.merge(data, actions, on=["userid"], how="left")
        data["last_order_time_distance"] = data.apply(
                lambda x : time_distance(x["orderTime"], x["actionTime"]), axis=1)
        features = data[["userid", "last_order_time_distance"]]
        features.to_csv(path, index=False)
    return features

def get_last_action_type_feature(tag="train") :
    """
    features                        comment
    first_action_type          上一次actionType
    """
    path = "cache/%s_last_action_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv("cache/train_action_series.csv")
        else :
            data = pd.read_csv("cache/test_action_series.csv")
        data = data.sort_values(by=["time"])
        data = data.drop_duplicates(["userid"], keep="last")
        features = data[["userid", "type"]]
        features = features.rename(columns={"type" : "last_action_type"})
        features.to_csv(path, index=False)
    return features

def get_first_action_type_feature(tag="train") :
    """
    features                        comment
    last_action_type          上一次actionType
    """
    path = "cache/%s_first_action_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        data = data.sort_values(by=["actionTime"])
        data = data.drop_duplicates(["userid"], keep="first")
        features = data[["userid", "actionType"]]
        features = features.rename(columns={"actionType" : "first_action_type"})
        features.to_csv(path, index=False)
    return features

def get_last_but_one_action_type_feature(tag="train") :
    """
    features                        comment
    last_but_one_action_type          倒数第二actionType
    """
    path = "cache/%s_last_but_one_action_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv("cache/train_action_series.csv")
        else :
            data = pd.read_csv("cache/test_action_series.csv")
        data = data.sort_values(by=["time"])
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        data = data.drop_duplicates(["userid"], keep="last")
        features = data[["userid", "type"]]
        features = features.rename(columns={"type" : "last_but_one_action_type"})
        features.to_csv(path, index=False)
    return features

def get_last_but_two_action_type_feature(tag="train") :
    """
    features                        comment
    last_but_two_action_type          倒数第三actionType
    """
    path = "cache/%s_last_but_two_action_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv("cache/train_action_series.csv")
        else :
            data = pd.read_csv("cache/test_action_series.csv")
        data = data.sort_values(by=["time"])
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        data = data.drop_duplicates(["userid"], keep="last")
        features = data[["userid", "type"]]
        features = features.rename(columns={"type" : "last_but_two_action_type"})
        features.to_csv(path, index=False)
    return features

def get_last_but_three_action_type_feature(tag="train") :
    """
    features                        comment
    last_but_three_action_type          倒数第四actionType
    """
    path = "cache/%s_last_but_three_action_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv("cache/train_action_series.csv")
        else :
            data = pd.read_csv("cache/test_action_series.csv")
        data = data.sort_values(by=["time"])
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        idx = data.duplicated(["userid"], keep="last")
        data = data[idx]
        data = data.drop_duplicates(["userid"], keep="last")
        features = data[["userid", "type"]]
        features = features.rename(columns={"type" : "last_but_three_action_type"})
        features.to_csv(path, index=False)
    return features

def get_history_order_count_features(tag="train") :
    """
    features            comment
    total_order_count  总历史订单数
    common_order_count 普通历史订单数
    special_order_count 精美历史订单数
    """
    path = "cache/%s_history_order_count.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(history_train_path)
        else :
            data = pd.read_csv(history_test_path)
        # get the total number of history orders
        total_count = data.groupby(['userid'], as_index=False).count()[["userid", "orderid"]]
        total_count.rename(columns={"orderid" : "total_order_count"}, inplace=True)
        # get the common number(orderType == 0) of history orders
        common_count = data[data["orderType"] == 0].groupby(['userid', 'orderType'], as_index=False).count()[["userid", "orderid"]]
        common_count.rename(columns={"orderid" : "common_order_count"}, inplace=True)
        # get the special number(orderType == 1) of history orders
        special_count = data[data["orderType"] == 1].groupby(['userid', 'orderType'], as_index=False).count()[["userid", "orderid"]]
        special_count.rename(columns={"orderid" : "special_order_count"}, inplace=True)
        # merge the three number 
        features = pd.merge(total_count, common_count, on=['userid'], how='left')
        features = pd.merge(features, special_count, on=["userid"], how='left')
        features = features.fillna(0)
        features["has_ordered_special"] = features["special_order_count"].map(lambda x : 1 if x > 0 else 0)
        features.to_csv(path, index=False)
    return features


def get_history_order_rate_features(tag="train") :
    """
    features            comment
    total_order_count  总历史订单数
    common_order_count 普通历史订单数
    special_order_count 精美历史订单数
    """
    path = "cache/%s_history_order_rate.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        data = pd.read_csv("cache/%s_history_order_count.csv" % tag)
        print data["special_order_count"]
        print data["common_order_count"]
        data["special_order_rate"] = data["special_order_count"] / [x if x != 0 else 1.0 for x in (data["special_order_count"] \
                + data["common_order_count"])]
        features = data[["userid", "special_order_rate"]]
        features.to_csv(path, index=False)
    return features

def get_action_count_features(tag="train") :
    """
    features            comment
    action_i        动作i的历史总数
    """
    path = "cache/%s_action_count.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(action_train_path)
        else :
            data = pd.read_csv(action_test_path)
        df = pd.get_dummies(data['actionType'], prefix='action')
        features = pd.concat([data, df], axis=1) 
        features = features.groupby(['userid'], as_index=False).sum()
        del features["actionType"]
        features.to_csv(path, index=False)
    return features

def get_action_rate_features(tag="train") :
    """
    features            comment
    action_i_ratio     动作i的历史总数占比
    """
    path = "cache/%s_action_rate.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv("cache/train_action_count.csv")
        else :
            data = pd.read_csv("cache/test_action_count.csv")
        data["total_action_count"] = data["action_1"] + data["action_2"] + data["action_3"] \
                                     + data["action_4"] + data["action_5"] + data["action_6"] \
                                     + data["action_7"] + data["action_8"] + data["action_9"]
        data['action_1_ratio'] = data['action_1'] / data['total_action_count']
        data['action_2_ratio'] = data['action_2'] / data['total_action_count']
        data['action_3_ratio'] = data['action_3'] / data['total_action_count']
        data['action_4_ratio'] = data['action_4'] / data['total_action_count']
        data['action_5_ratio'] = data['action_5'] / data['total_action_count']
        data['action_6_ratio'] = data['action_6'] / data['total_action_count']
        data['action_7_ratio'] = data['action_7'] / data['total_action_count']
        data['action_8_ratio'] = data['action_8'] / data['total_action_count']
        data['action_9_ratio'] = data['action_9'] / data['total_action_count']
        features = data[["action_1_ratio", "action_2_ratio", "action_3_ratio",
                         "action_4_ratio", "action_5_ratio", "action_6_ratio",
                         "action_7_ratio", "action_8_ratio", "action_9_ratio",
                         "total_action_count", "userid"]]
        features.to_csv(path, index=False)
    return features

def get_action_rate_from_last_order_features(tag="train") :
    """
    features                                comment
    action_i_ratio_from_last_order     动作i的历史总数占比自从上次订单
    """
    path = "cache/%s_action_rate_from_last_order.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv("cache/train_action_count_from_last_order.csv")
        else :
            data = pd.read_csv("cache/test_action_count_from_last_order.csv")
        data["total_action_count_from_last_order"] = data["action_from_last_order_1"] \
                                                   + data["action_from_last_order_2"] \
                                                   + data["action_from_last_order_3"] \
                                                   + data["action_from_last_order_4"] \
                                                   + data["action_from_last_order_5"] \
                                                   + data["action_from_last_order_6"] \
                                                   + data["action_from_last_order_7"] \
                                                   + data["action_from_last_order_8"] \
                                                   + data["action_from_last_order_9"] 
        data['action_1_ratio_from_last_order'] = data['action_from_last_order_1'] / data['total_action_count_from_last_order']
        data['action_2_ratio_from_last_order'] = data['action_from_last_order_2'] / data['total_action_count_from_last_order']
        data['action_3_ratio_from_last_order'] = data['action_from_last_order_3'] / data['total_action_count_from_last_order']
        data['action_4_ratio_from_last_order'] = data['action_from_last_order_4'] / data['total_action_count_from_last_order']
        data['action_5_ratio_from_last_order'] = data['action_from_last_order_5'] / data['total_action_count_from_last_order']
        data['action_6_ratio_from_last_order'] = data['action_from_last_order_6'] / data['total_action_count_from_last_order']
        data['action_7_ratio_from_last_order'] = data['action_from_last_order_7'] / data['total_action_count_from_last_order']
        data['action_8_ratio_from_last_order'] = data['action_from_last_order_8'] / data['total_action_count_from_last_order']
        data['action_9_ratio_from_last_order'] = data['action_from_last_order_9'] / data['total_action_count_from_last_order']
        features = data[["action_1_ratio_from_last_order", 
                         "action_2_ratio_from_last_order", 
                         "action_3_ratio_from_last_order",
                         "action_4_ratio_from_last_order", 
                         "action_5_ratio_from_last_order", 
                         "action_6_ratio_from_last_order",
                         "action_7_ratio_from_last_order", 
                         "action_8_ratio_from_last_order", 
                         "action_9_ratio_from_last_order",
                         "total_action_count_from_last_order", 
                         "userid"]]
        features.to_csv(path, index=False)
    return features
def get_action_count_from_last_order(tag="train") :
    """
    features                        comment
    action_from_last_order_i     最后一个订单后各种action的数目
    """
    path = "cache/%s_action_count_from_last_order.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            actions = pd.read_csv(action_train_path)
            orders = pd.read_csv(history_train_path)
        else :
            actions = pd.read_csv(action_test_path)
            orders = pd.read_csv(history_test_path)
        orders = orders.sort_values(by=["orderTime"])
        orders = orders.drop_duplicates(["userid"], keep="last")
        data = pd.merge(actions, orders, on=["userid"], how="left")
        data["orderTime"].replace(np.nan, early_time, inplace=True)
        #data = data[data["orderTime"].notnull()]
        data = data[data["actionTime"] > data["orderTime"]]
        df = pd.get_dummies(data["actionType"], prefix="action_from_last_order")
        data = data[["userid"]]
        features = pd.concat([data, df], axis=1)
        features = features.groupby(["userid"], as_index=False).sum()
        features.to_csv(path, index=False)
    return features

def get_last_ordertype_features(tag="train") :
    """
    features            comment
    last_order_type     历史最后订单类型
    """
    path = "cache/%s_last_ordertype.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(history_train_path)
        else :
            data = pd.read_csv(history_test_path)
        data = data.sort_values(by=["orderTime"])
        data = data.drop_duplicates(["userid"], keep="last")
        features = data[["userid", "orderType"]]
        features.rename(columns={"orderType" : "last_order_type"}, inplace=True)
        features.to_csv(path, index=False)
    return features

def get_comment_features(tag="train") :
    """
    features            comment
    has_comment     是否有评论
    has_tag         是否有标记
    """
    path = "cache/%s_comment.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(comment_train_path)
        else :
            data = pd.read_csv(comment_test_path)
        data["has_comment"] = (~data["commentsKeyWords"].isnull()).astype('int')
        data["has_tag"] = (~data["tags"].isnull()).astype('int')
        features = data[["userid", "rating", "has_comment", "has_tag"]]
        features.to_csv(path, index=False)
    return features

def get_comment_num_features(tag="train") :
    """
    features            comment
    comment_num     评论的关键词数目
    tag_num         标记的数目
    """
    path = "cache/%s_comment_num.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(comment_train_path)
        else :
            data = pd.read_csv(comment_test_path)
        data["comment_num"] = data["commentsKeyWords"].fillna("[]").astype('string').map(lambda x : len(eval(x)))
        data["tag_num"] = data["tags"].fillna("").astype('string').map(lambda x : len(x.split('|')) if x != "" else 0)
        features = data[["userid", "tag_num", "comment_num"]]
        features.to_csv(path, index=False)
    return features

def get_comment_time_distance(tag="train") :
    '''
    features                    comment
    comment_time_distance       评论的订单时间间隔
    '''
    path = "cache/%s_comment_time_distance.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            actions = pd.read_csv(action_train_path)
            orders = pd.read_csv(history_train_path)
            comments = pd.read_csv(comment_train_path)
        else :
            actions = pd.read_csv(action_test_path)
            orders = pd.read_csv(history_test_path)
            comments = pd.read_csv(comment_test_path)
        last = actions.sort_values(by=["actionTime"])
        last.drop_duplicates(["userid"], keep="last", inplace=True)
        last = last[["userid", "actionTime"]]
        last.rename(columns={"actionTime" : "last_time"}, inplace=True)

        comments = pd.merge(comments, orders, on=["userid", "orderid"], how="inner")
        comments = comments[["userid", "orderTime"]]
        features = pd.merge(comments, last, on=["userid"], how="left")
        features["comment_time_distance"] =  features.apply( \
                lambda x : time_distance(x["orderTime"], x["last_time"]), axis=1)
        features = features[["userid", "comment_time_distance"]]
        features.to_csv(path, index=False)
    return features

def get_history_destination_features(tag="train") :
    '''
    features                    comment
    city_diversity          去过的城市的数目
    country_diversity       去过的国家的数目
    continent_diversity     去过的大陆的数目 
    '''
    path = "cache/%s_history_destination.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(history_train_path)
        else :
            data = pd.read_csv(history_test_path)
        features = pd.DataFrame(data["userid"].unique())
        features.columns = ["userid"]
        data = data.groupby(["userid"], as_index=False)
        features["city_diversity"] = data.apply(lambda x : len(x["city"].unique()))
        features["country_diversity"] = data.apply(lambda x : len(x["country"].unique()))
        features["continent_diversity"] = data.apply(lambda x : len(x["continent"].unique()))
        features.to_csv(path, index=False)
    return features

def get_continent_features(tag="train") :
    '''
    features                    comment
    continent_i          去过大陆i的数目
    '''
    path = "cache/%s_continent.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(history_train_path)
        else :
            data = pd.read_csv(history_test_path)
        df = pd.get_dummies(data['continent'], prefix='continent')
        features = pd.concat([data, df], axis=1) 
        features = features.groupby(['userid'], as_index=False).sum()
        features = features[["userid", "continent_0", "continent_1", "continent_2", "continent_3",
                             "continent_4", "continent_5"]]
        features.to_csv(path, index=False)
    return features

def get_country_features(tag="train") :
    '''
    features                    comment
    country_i          去过国家i的数目
    '''
    path = "cache/%s_country.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            data = pd.read_csv(history_train_path)
        else :
            data = pd.read_csv(history_test_path)
        df = pd.get_dummies(data['country'], prefix='country')
        features = pd.concat([data, df], axis=1) 
        features = features.groupby(['userid'], as_index=False).sum()
        set = ["country_%d" % i for i in range(0,51)]
        set.append("userid")
        features = features[set]
        features.to_csv(path, index=False)
    return features

def get_order_rate_by_last_type_features(tag="train") :
    '''
    features                    comment
    order_rate_by_last_type     最后一个actionType各自的购买精美服务率
    '''
    path = "cache/%s_order_rate_by_last_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            actions = pd.read_csv(action_train_path)
            futures = pd.read_csv(future_train_path)
        else :
            actions = pd.read_csv(action_test_path)
            futures = pd.read_csv(future_test_path)
        actions.sort_values(by=["actionTime"], inplace=True)
        actions.drop_duplicates(["userid"], keep="last", inplace=True)
        data = pd.merge(futures, actions, on=["userid"], how="left")
        data["actionType"].fillna(0, inplace=True)
        df = pd.get_dummies(data["orderType"], prefix="orderType")
        data = pd.concat([data, df], axis=1)
        data = data.groupby(["actionType"]).sum()
        features = data[["orderType_0", "orderType_1"]]
        features = features.reset_index()
        features["order_rate_by_last_type"] = \
            features["orderType_1"] / (features["orderType_1"] + features["orderType_0"])
        features = features[["actionType", "order_rate_by_last_type"]]
        features.rename(columns={"actionType" : "last_action_type"}, inplace=True)
        features.to_csv(path, index=False)
    return features

def get_order_rate_by_age_and_last_type_features(tag="train") :
    '''
    features                    comment
    order_rate_by_age_and_last_type     最后一个actionType和年龄各自的购买精美服务率
    '''
    path = "cache/%s_order_rate_by_age_and_last_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            users = pd.read_csv(user_train_path)
            actions = pd.read_csv(action_train_path)
            futures = pd.read_csv(future_train_path)
        else :
            users = pd.read_csv(user_test_path)
            actions = pd.read_csv(action_test_path)
            futures = pd.read_csv(future_test_path)
        actions.sort_values(by=["actionTime"], inplace=True)
        actions.drop_duplicates(["userid"], keep="last", inplace=True)
        data = pd.merge(futures, actions, on=["userid"], how="left")
        data["actionType"].fillna(0, inplace=True)
        df = pd.get_dummies(data["orderType"], prefix="orderType")
        data = pd.concat([data, df], axis=1)
        data = pd.merge(data, users, on=["userid"], how="left")
        data = data.groupby(["actionType", "age"]).sum()
        features = data[["orderType_0", "orderType_1"]]
        features = features.reset_index()
        features["order_rate_by_age_and_last_type"] = \
            features["orderType_1"] / (features["orderType_1"] + features["orderType_0"])
        features = features[["actionType", "age", "order_rate_by_age_and_last_type"]]
        features.rename(columns={"actionType" : "last_action_type"}, inplace=True)
        features.to_csv(path, index=False)
    return features

def get_order_rate_by_age_features(tag="train") :
    '''
    features                    comment
    order_rate_by_age     各个年龄段的购买精美服务率
    '''
    path = "cache/%s_order_rate_by_age.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            users = pd.read_csv(user_train_path)
            futures = pd.read_csv(future_train_path)
        else :
            users = pd.read_csv(user_test_path)
            futures = pd.read_csv(future_test_path)
        data = pd.merge(futures, users, on=["userid"], how="left")
        df = pd.get_dummies(data["orderType"], prefix="orderType")
        data = pd.concat([data, df], axis=1)
        features = data.groupby(["age"]).sum()
        features = features[["orderType_1", "orderType_0"]]
        features.reset_index(inplace=True)
        features["order_rate_by_age"] = \
            features["orderType_1"] / (features["orderType_1"] + features["orderType_0"])
        features = features[["age", "order_rate_by_age"]]
        features.to_csv(path, index=False)
    return features

def get_order_rate_by_gender_and_age_features(tag="train") :
    '''
    features                            comment
    order_rate_by_gender_and_age     各个年龄段和各个性别的购买精美服务率
    '''
    path = "cache/%s_order_rate_by_gender_and_age.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            users = pd.read_csv(user_train_path)
            futures = pd.read_csv(future_train_path)
        else :
            users = pd.read_csv(user_test_path)
            futures = pd.read_csv(future_test_path)
        data = pd.merge(futures, users, on=["userid"], how="left")
        df = pd.get_dummies(data["orderType"], prefix="orderType")
        data = pd.concat([data, df], axis=1)
        features = data.groupby(["age", "gender"]).sum()
        features = features[["orderType_1", "orderType_0"]]
        features.reset_index(inplace=True)
        features["order_rate_by_gender_and_age"] = \
            features["orderType_1"] / (features["orderType_1"] + features["orderType_0"])
        features = features[["age", "gender", "order_rate_by_gender_and_age"]]
        features.to_csv(path, index=False)
    return features

def get_order_rate_by_gender_and_age_and_province_features(tag="train") :
    '''
    features                                        comment
    order_rate_by_gender_and_age_and_province     各个年龄段和各个性别,省份的购买精美服务率
    '''
    path = "cache/%s_order_rate_by_gender_and_age_and_province.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            users = pd.read_csv(user_train_path)
            futures = pd.read_csv(future_train_path)
        else :
            users = pd.read_csv(user_test_path)
            futures = pd.read_csv(future_test_path)
        data = pd.merge(futures, users, on=["userid"], how="left")
        df = pd.get_dummies(data["orderType"], prefix="orderType")
        data = pd.concat([data, df], axis=1)
        features = data.groupby(["age", "gender", "province"]).sum()
        features = features[["orderType_1", "orderType_0"]]
        features.reset_index(inplace=True)
        features["order_rate_by_gender_and_age_and_province"] = \
            features["orderType_1"] / (features["orderType_1"] + features["orderType_0"])
        features = features[["age", "gender", "province", "order_rate_by_gender_and_age_and_province"]]
        features.to_csv(path, index=False)
    return features

def get_order_rate_by_last_two_type_features(tag="train") :
    '''
    features                    comment
    order_rate_by_last_two_type     最后两个actionType各自的购买精美服务率
    '''
    path = "cache/%s_order_rate_by_last_two_type.csv" % tag
    if os.path.exists(path) :
        features = pd.read_csv(path)
    else :
        if tag == "train" :
            actions = pd.read_csv(action_train_path)
            futures = pd.read_csv(future_train_path)
        else :
            actions = pd.read_csv(action_test_path)
            futures = pd.read_csv(future_test_path)
        actions = actions.sort_values(by=["actionTime"])
        last_actions = actions.drop_duplicates(["userid"], keep="last")
        last_actions.rename(columns={"actionType" : "last_action_type"}, inplace=True)
        idx = actions.duplicated(["userid"], keep="last")
        last_but_one_actions = actions[idx]
        last_but_one_actions.drop_duplicates(["userid"], keep="last", inplace=True)
        last_but_one_actions.rename(columns={"actionType" : "last_but_one_action_type"}, inplace=True)
        data = pd.merge(futures, last_actions, on=["userid"], how="left")
        data = pd.merge(data, last_but_one_actions, on=["userid"], how="left")
        data["last_action_type"].fillna(0, inplace=True)
        data["last_but_one_action_type"].fillna(0, inplace=True)
        df = pd.get_dummies(data["orderType"], prefix="orderType")
        data = pd.concat([data, df], axis=1)
        data = data.groupby(["last_action_type", "last_but_one_action_type"]).sum()
        features = data[["orderType_0", "orderType_1"]]
        features = features.reset_index()
        features["order_rate_by_last_two_type"] = \
            features["orderType_1"] / (features["orderType_1"] + features["orderType_0"])
        features = features[["last_action_type", "last_but_one_action_type", "order_rate_by_last_two_type"]]
        features.to_csv(path, index=False)
    return features
    
def make_train_set() :
    data = pd.read_csv(user_train_path)
    data = pd.merge(data, get_history_order_count_features(), on=["userid"], how='left')
    data = pd.merge(data, get_action_count_features(), on=["userid"], how='left')
    data = pd.merge(data, get_action_rate_features(), on=["userid"], how='left')
    data = pd.merge(data, pd.read_csv(future_train_path), on=["userid"], how='left')
    data = pd.merge(data, get_last_special_order_time_distance_features(), on=["userid"], how="left")
    data = pd.merge(data, get_last_action_type_feature(), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_one_action_type_feature(), on=["userid"], how="left")
    data = pd.merge(data, get_special_order_time_distance_mean_features(), on=["userid"], how="left")
    data = pd.merge(data, get_action_time_distance_features(), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_two_action_type_feature(), on=["userid"], how="left")
    data = pd.merge(data, get_last_ordertype_features(), on=["userid"], how="left")
    data = pd.merge(data, get_action_count_from_last_order(), on=["userid"], how="left")
    data = pd.merge(data, get_last_two_action_time_distance_features(), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_three_action_type_feature(), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_two_action_time_distance_features(), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_three_action_time_distance_features(), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_four_action_time_distance_features(), on=["userid"], how="left")
    data = pd.merge(data, get_last_four_time_distance_mean(), on=["userid"], how="left")
    data = pd.merge(data, get_last_four_time_distance_std(), on=["userid"], how="left")
    data = pd.merge(data, get_time_distance_from_first_actiontype_features(), on=["userid"], how="left")
    data = pd.merge(data, get_history_order_rate_features(), on=["userid"], how="left")
    data = pd.merge(data, get_comment_features(), on=["userid"], how="left")
    data = pd.merge(data, get_comment_time_distance(), on=["userid"], how="left")
    data = pd.merge(data, get_last_four_action_type_string_features(), on=["userid"], how="left")
    data = pd.merge(data, get_country_features(), on=["userid"], how="left")
    data = pd.merge(data, get_history_destination_features(), on=["userid"], how="left")
    data = pd.merge(data, get_continent_features(), on=["userid"], how="left")
    data = pd.merge(data, get_comment_num_features(), on=["userid"], how="left")
    data = pd.merge(data, get_first_action_type_feature(), on=["userid"], how="left")
    data = pd.merge(data, get_last_time_hour(), on=["userid"], how="left")
    data["hour"].fillna(25, inplace=True)
    data = pd.merge(data, get_last_action_continuous_count_features(600), on=["userid"], how="left")
    data = pd.merge(data, get_last_time_day(), on=["userid"], how="left")
    data = pd.merge(data, get_action_rate_from_last_order_features(), on=["userid"], how="left")
    data = pd.merge(data, get_last_order_time_distance_features(), on=["userid"], how="left")
    for i in range(1,10) :
        data = pd.merge(data, get_time_distance_from_last_actionType_features(i), on=["userid"], how="left")
        data = pd.merge(data, get_action_type_time_distance_features(i), on=["userid"], how="left")
    for time in [1440, 9080] :
        data = pd.merge(data, get_special_time_action_count_features(time), on=["userid"], how="left")
    data = data.fillna(0)
    data = pd.merge(data, get_order_rate_by_last_two_type_features(), 
            on=["last_action_type", "last_but_one_action_type"], how="left")
    return data

def make_test_set() :
    data = pd.read_csv(user_test_path)
    data = pd.merge(data, get_history_order_count_features(tag="test"), on=["userid"], how='left')
    data = pd.merge(data, get_action_count_features("test"), on=["userid"], how='left')
    data = pd.merge(data, get_action_rate_features("test"), on=["userid"], how='left')
    data = pd.merge(data, get_last_special_order_time_distance_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_action_type_feature("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_one_action_type_feature("test"), on=["userid"], how="left")
    data = pd.merge(data, get_special_order_time_distance_mean_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_action_time_distance_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_two_action_type_feature("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_ordertype_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_action_count_from_last_order("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_two_action_time_distance_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_three_action_type_feature("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_two_action_time_distance_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_three_action_time_distance_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_but_four_action_time_distance_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_four_time_distance_mean("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_four_time_distance_std("test"), on=["userid"], how="left")
    data = pd.merge(data, get_time_distance_from_first_actiontype_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_history_order_rate_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_comment_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_comment_time_distance("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_four_action_type_string_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_continent_features("test"), on=["userid"], how="left")
    #data = pd.merge(data, get_country_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_history_destination_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_comment_num_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_first_action_type_feature("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_time_hour("test"), on=["userid"], how="left")
    data["hour"].fillna(25, inplace=True)
    data = pd.merge(data, get_last_action_continuous_count_features(600, "test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_time_day("test"), on=["userid"], how="left")
    data = pd.merge(data, get_action_rate_from_last_order_features("test"), on=["userid"], how="left")
    data = pd.merge(data, get_last_order_time_distance_features("test"), on=["userid"], how="left")
    for i in range(1,10) :
        data = pd.merge(data, get_time_distance_from_last_actionType_features(i, "test"), on=["userid"], how="left")
        data = pd.merge(data, get_action_type_time_distance_features(i, "test"), on=["userid"], how="left")
    for time in [1440, 9080] :
        data = pd.merge(data, get_special_time_action_count_features(time, "test"), on=["userid"], how="left")
    data = data.fillna(0)
    data = pd.merge(data, get_order_rate_by_last_two_type_features("test"), 
            on=["last_action_type", "last_but_one_action_type"], how="left")
    return data

if __name__ == "__main__" :
    #data = make_train_set()
    #print data[["orderType", "last_action_type", "last_but_one_action_type", "last_but_two_action_type"]]
    #data.to_csv("cache/train_features.csv", index=False)
    #get_special_time_action_count_features(10)
    #print get_action_time_distance_features()
    data = pd.read_csv(user_train_path)
    data = pd.merge(data, get_history_order_count_features(), on=["userid"], how='left')
    data = data[data["special_order_count"] == 0]
