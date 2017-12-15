
# Dec 5, 2017, XyZ

import numpy as np
import pandas as pd
import datetime

data_org = pd.read_csv('tianchi_fresh_comp_train_user.csv')
item_org = pd.read_csv('tianchi_fresh_comp_train_item.csv')

data_org['time'] = pd.to_datetime(data_org['time'])
data_org['date'] = data_org['time'].dt.date
data_org['hour'] = data_org['time'].dt.hour

def handle1212():
    purchased = (data_org.behavior_type == 4)
    # Exclude users who only purchased on 2014/12/12
    is_1212 = (data_org.date == datetime.date(2014, 12, 12))
    is_not_1212 = (data_org.date != datetime.date(2014, 12, 12))

    purchased_users_1212 = data_org[is_1212 & purchased].user_id.drop_duplicates()
    purchased_exclude_1212 = data_org[is_not_1212 & purchased]

    solo_user_1212 = []
    for user in purchased_users_1212:
        if user not in purchased_exclude_1212.user_id:
            solo_user_1212.append(user)

    exclude_SoloUser_1212 = ~data_org.user_id.isin(solo_user_1212)
    return data_org[exclude_SoloUser_1212]

newdata = handle1212()

def data_in_Window(dateSet, date, num):
    startDate = date - datetime.timedelta(num)
    endDate = date - datetime.timedelta(1)
    is_inDateWindow = (dateSet.date >= startDate) & (dateSet.date <= endDate)
    return dateSet[is_inDateWindow]

def get_windows(date):
    historyData = newdata[newdata.date < date]
    ytd_data = data_in_Window(historyData, date, 1)
    data_in2days = data_in_Window(historyData, date, 2)
    data_in3days = data_in_Window(historyData, date, 3)
    data_in4days = data_in_Window(historyData, date, 4)
    data_in5days = data_in_Window(historyData, date, 5)
    data_inOneweek = data_in_Window(historyData, date, 7)
    data_inTwoweek = data_in_Window(historyData, date, 14)
    return ytd_data, data_in2days, data_in3days, data_in4days, data_in5days, data_inOneweek, data_inTwoweek

def get_ctr(dateSet):
    dateSet['ctr41'] = dateSet[4]/dateSet[1]
    dateSet['ctr24'] = dateSet[2]/dateSet[4]
    dateSet['ctr43'] = dateSet[4]/dateSet[3]
    return dateSet.fillna(0)

def get_ui_features(date):
    # Inspect those buy with prebehaviors
    ytdData, dataIn2days, dataIn3days, dataIn4days, dataIn5days, dataIn1week, dataIn2weeks = get_windows(date)
    
    lovedHistory = dataIn2weeks[dataIn2weeks.behavior_type == 2]
    purchasedHistory = dataIn2weeks[dataIn2weeks.behavior_type == 4]
    
    ui_actions_ytd = pd.crosstab([ytdData.user_id, ytdData.item_id], ytdData.behavior_type)
    ui_actions_ytd = get_ctr(ui_actions_ytd)
    ui_actions_ytd.columns = ['yui1', 'yui2', 'yui3', 'yui4', 'yui_ctr41', 'yui_ctr24', 'yui_ctr43']
    ui_actions_within2 = pd.crosstab([dataIn2days.user_id, dataIn2days.item_id], dataIn2days.behavior_type)
    ui_actions_within2 = get_ctr(ui_actions_within2)
    ui_actions_within2.columns = ['ui1_in2', 'ui2_in2', 'ui3_in2', 'ui4_in2', 'ui2_ctr41', 'ui2_ctr24', 'ui2_ctr43']
    ui_actions_within3 = pd.crosstab([dataIn3days.user_id, dataIn3days.item_id], dataIn3days.behavior_type)
    ui_actions_within3 = get_ctr(ui_actions_within3)
    ui_actions_within3.columns = ['ui1_in3', 'ui2_in3', 'ui3_in3', 'ui4_in3', 'ui3_ctr41', 'ui3_ctr24', 'ui3_ctr43']
    ui_actions_within4 = pd.crosstab([dataIn4days.user_id, dataIn4days.item_id], dataIn4days.behavior_type)
    ui_actions_within4 = get_ctr(ui_actions_within4)
    ui_actions_within4.columns = ['ui1_in4', 'ui2_in4', 'ui3_in4', 'ui4_in4', 'ui4_ctr41', 'ui4_ctr24', 'ui4_ctr43']
    ui_actions_within5 = pd.crosstab([dataIn5days.user_id, dataIn5days.item_id], dataIn5days.behavior_type)
    ui_actions_within5 = get_ctr(ui_actions_within5)
    ui_actions_within5.columns = ['ui1_in5', 'ui2_in5', 'ui3_in5', 'ui4_in5', 'ui5_ctr41', 'ui5_ctr24', 'ui5_ctr43']
    
    ui_buy_total = pd.crosstab([purchasedHistory.user_id, purchasedHistory.item_id], purchasedHistory.behavior_type)
    ui_buy_total.columns = ['ui_B']

    ui_loved_total = pd.crosstab([lovedHistory.user_id, lovedHistory.item_id], lovedHistory.behavior_type)
    ui_loved_total.columns = ['ui_l']
    ui_loved_total.ui_loved = 1
    
    ui_touch_days = dataIn2weeks.groupby(['user_id','item_id']).agg({"date":lambda x:(x.max()-x.min())})
	
    ui_History = pd.concat([ui_loved_total, ui_buy_total, ui_touch_days, ui_actions_ytd, 
        ui_actions_within3, ui_actions_within5], axis=1, join_axes=[ui_actions_ytd.index]).fillna(0)
    return ui_History

def get_uc_features(date):
    # Inspect those buy with prebehaviors
    ytdData, dataIn2days, dataIn3days, dataIn4days, dataIn5days, dataIn1week, dataIn2weeks = get_windows(date)
    
    uc_actions_ytd = pd.crosstab([ytdData.user_id, ytdData.item_category], ytdData.behavior_type)
    uc_actions_ytd = get_ctr(uc_actions_ytd)
    uc_actions_ytd.columns = ['yuc1', 'yuc2', 'yuc3', 'yuc4', 'yuc_ctr41', 'yuc_ctr24', 'yuc_ctr43']
    uc_actions_within2 = pd.crosstab([dataIn2days.user_id, dataIn2days.item_category], dataIn2days.behavior_type)
    uc_actions_within2 = get_ctr(uc_actions_within2)
    uc_actions_within2.columns = ['uc1_in2', 'uc2_in2', 'uc3_in2', 'uc4_in2', 'uc2_ctr41', 'uc2_ctr24', 'uc2_ctr43']
    uc_actions_within3 = pd.crosstab([dataIn3days.user_id, dataIn3days.item_category], dataIn3days.behavior_type)
    uc_actions_within3 = get_ctr(uc_actions_within3)
    uc_actions_within3.columns = ['uc1_in3', 'uc2_in3', 'uc3_in3', 'uc4_in3', 'uc3_ctr41', 'uc3_ctr24', 'uc3_ctr43']
    uc_actions_within4 = pd.crosstab([dataIn4days.user_id, dataIn4days.item_category], dataIn4days.behavior_type)
    uc_actions_within4 = get_ctr(uc_actions_within4)
    uc_actions_within4.columns = ['uc1_in4', 'uc2_in4', 'uc3_in4', 'uc4_in4', 'uc4_ctr41', 'uc4_ctr24', 'uc4_ctr43']
    uc_actions_within5 = pd.crosstab([dataIn5days.user_id, dataIn5days.item_category], dataIn5days.behavior_type)
    uc_actions_within5 = get_ctr(uc_actions_within5)
    uc_actions_within5.columns = ['uc1_in5', 'uc2_in5', 'uc3_in5', 'uc4_in5', 'uc5_ctr41', 'uc5_ctr24', 'uc5_ctr43']
    
    uc_touch_days = dataIn2weeks.groupby(['user_id','item_category']).agg({"date":lambda x:(x.max()-x.min())})
    
    uc_History = pd.concat([uc_touch_days, uc_actions_ytd, uc_actions_within2, uc_actions_within3, 
        uc_actions_within4, uc_actions_within5], axis=1, join_axes=[uc_actions_ytd.index]).fillna(0)
    #uc_History.columns = ['yuc1', 'yuc2', 'yuc3', 'yuc4', 'uc1_in3', 'uc2_in3', 'uc3_in3', 'uc4_in3', 'uc1_in5', 'uc2_in5', 'uc3_in5', 'uc4_in5']
    return uc_History

def get_user_feature(date):
    # Inspect history purchased users
    ytdData, dataIn2days, dataIn3days, dataIn4days, dataIn5days, dataIn1week, dataIn2weeks = get_windows(date)
    
    ytd_user_actions = pd.crosstab(ytdData.user_id, ytdData.behavior_type)
    ytd_user_actions = get_ctr(ytd_user_actions)
    ytd_user_actions.columns = ['yu1', 'yu2', 'yu3', 'yu4', 'yu_ctr41', 'yu_ctr24', 'yu_ctr43']
    user_actions_in2days = pd.crosstab(dataIn2days.user_id, dataIn2days.behavior_type)
    user_actions_in2days = get_ctr(user_actions_in2days)
    user_actions_in2days.columns = ['u1_in2', 'u2_in2', 'u3_in2', 'u4_in2', 'u2_ctr41', 'u2_ctr24', 'u2_ctr43']
    user_actions_in3days = pd.crosstab(dataIn3days.user_id, dataIn3days.behavior_type)
    user_actions_in3days = get_ctr(user_actions_in3days)
    user_actions_in3days.columns = ['u1_in3', 'u2_in3', 'u3_in3', 'u4_in3', 'u3_ctr41', 'u3_ctr24', 'u3_ctr43']
    #user_actions_daily = pd.crosstab([dataIn2weeks.user_id, dataIn2weeks.behavior_type], dataIn2weeks.date)
    #ua_daily = user_actions_daily.applymap(lambda x:1 if x>0 else 0).sum()
    
    user_records = pd.concat([ytd_user_actions, user_actions_in2days, user_actions_in3days], axis=1, join_axes=[ytd_user_actions.index]).fillna(0)
    #user_records.columns = ['yu1', 'yu2', 'yu3', 'yu4', 'u1_in3', 'u2_in3', 'u3_in3', 'u4_in3', 'u1_in5', 'u2_in5', 'u3_in5', 'u4_in5']
    return user_records

def get_item_feature(date):
    # Inspect history sold items
    ytdData, dataIn2days, dataIn3days, dataIn4days, dataIn5days, dataIn1week, dataIn2weeks = get_windows(date)
    
    ytd_item_actions = pd.crosstab(ytdData.item_id, ytdData.behavior_type)
    ytd_item_actions = get_ctr(ytd_item_actions)
    ytd_item_actions.columns = ['yi1', 'yi2', 'yi3', 'yi4', 'yi_ctr41', 'yi_ctr24', 'yi_ctr43']
    item_actions_in2days = pd.crosstab(dataIn2days.item_id, dataIn2days.behavior_type)
    item_actions_in2days = get_ctr(item_actions_in2days)
    item_actions_in2days.columns = ['i1_in2', 'i2_in2', 'i3_in2', 'i4_in2', 'i2_ctr41', 'i2_ctr24', 'i2_ctr43']
    item_actions_in3days = pd.crosstab(dataIn3days.item_id, dataIn3days.behavior_type)
    item_actions_in3days = get_ctr(item_actions_in3days)
    item_actions_in3days.columns = ['i1_in3', 'i2_in3', 'i3_in3', 'i4_in3', 'i3_ctr41', 'i3_ctr24', 'i3_ctr43']

    item_records = pd.concat([ytd_item_actions, item_actions_in2days, item_actions_in3days], axis=1, join_axes=[ytd_item_actions.index]).fillna(0)
    #item_records.columns = ['yi1', 'yi2', 'yi3', 'yi4', 'i1_in3', 'i2_in3', 'i3_in3', 'i4_in3', 'i1_in5', 'i2_in5', 'i3_in5', 'i4_in5']
    return item_records

def get_category_feature(date):
    # Inspect history sold items_category
    ytdData, dataIn2days, dataIn3days, dataIn4days, dataIn5days, dataIn1week, dataIn2weeks = get_windows(date)
    
    ytd_cate_actions = pd.crosstab(ytdData.item_category, ytdData.behavior_type)
    ytd_cate_actions = get_ctr(ytd_cate_actions)
    ytd_cate_actions.columns = ['yc1', 'yc2', 'yc3', 'yc4', 'yc_ctr41', 'yc_ctr24', 'yc_ctr43']
    cate_actions_in2days = pd.crosstab(dataIn2days.item_category, dataIn2days.behavior_type)
    cate_actions_in2days = get_ctr(cate_actions_in2days)
    cate_actions_in2days.columns = ['c1_in2', 'c2_in2', 'c3_in2', 'c4_in2', 'c2_ctr41', 'c2_ctr24', 'c2_ctr43']
    cate_actions_in3days = pd.crosstab(dataIn3days.item_category, dataIn3days.behavior_type)
    cate_actions_in3days = get_ctr(cate_actions_in3days)
    cate_actions_in3days.columns = ['c1_in3', 'c2_in3', 'c3_in3', 'c4_in3', 'c3_ctr41', 'c3_ctr24', 'c3_ctr43']
    
    category_records = pd.concat([ytd_cate_actions, cate_actions_in2days, cate_actions_in3days], axis=1, join_axes=[ytd_cate_actions.index]).fillna(0)
    #category_records.columns = ['yc1', 'yc2', 'yc3', 'yc4', 'c1_in3', 'c2_in3', 'c3_in3', 'c4_in3', 'c1_in5', 'c2_in5', 'c3_in5', 'c4_in5']
    return category_records

def get_yesterday_uic(date):
    ytdData, dataIn2days, dataIn3days, dataIn4days, dataIn5days, dataIn1week, dataIn2weeks = get_windows(date)
    
    testUIC = ytdData[['user_id', 'item_id','item_category']].drop_duplicates(['user_id', 'item_id'])
    #ytd_ua_hourly = pd.crosstab([ytdData.user_id,ytdData.behavior_type], ytdData.hour, dropna=False)
    #ytd_ua_hourly = ytd_ua_hourly.unstack(fill_value = 0)

    #ytd_uia_hourly = pd.crosstab([ytdData.user_id,ytdData.item_id,ytdData.behavior_type], ytdData.hour, dropna=False)
    #ytd_uia_hourly = ytd_uia_hourly.unstack(fill_value = 0)

    #ytd_uca_hourly = pd.crosstab([ytdData.user_id,ytdData.item_category,ytdData.behavior_type], ytdData.hour, dropna=False)
    #ytd_uca_hourly = ytd_uca_hourly.unstack(fill_value = 0)

    #ua_hourly_in14 = pd.crosstab([dataIn2weeks.user_id,dataIn2weeks.behavior_type], dataIn2weeks.hour, dropna=False)
    #ua_hourly_in14 = ua_hourly_in14.unstack(fill_value = 0)
    #hourly_behaviors_count = pd.merge(ytd_UIC, ytd_ua_hourly, left_on=['user_id'], right_index=True, how='left')
    #hourly_behaviors_count = pd.merge(hourly_behaviors_count, ytd_uia_hourly, left_on=['user_id', 'item_id'], right_index=True, how='left')
    #hourly_behaviors_count = pd.merge(hourly_behaviors_count, ytd_uca_hourly, left_on=['user_id', 'item_category'], right_index=True, how='left')
    #hourly_behaviors_count = pd.merge(hourly_behaviors_count, ua_hourly_in14, left_on=['user_id'], right_index=True, how='left')
    return testUIC

def get_features(date):
    ytd_UIC = get_yesterday_uic(date)
    
    uiHistory = get_ui_features(date)
    ucHistory = get_uc_features(date)
    userRecords = get_user_feature(date)
    itemRecords = get_item_feature(date)
    categoryRecords = get_category_feature(date)
    
    feature_set = pd.merge(ytd_UIC, uiHistory, left_on=['user_id', 'item_id'], right_index=True, how='left')
    feature_set = pd.merge(feature_set, ucHistory, left_on=['user_id', 'item_category'], right_index=True, how='left')
    feature_set = pd.merge(feature_set, userRecords, left_on=['user_id'], right_index=True, how='left')
    feature_set = pd.merge(feature_set, itemRecords, left_on=['item_id'], right_index=True, how='left')
    feature_set = pd.merge(feature_set, categoryRecords, left_on=['item_category'], right_index=True, how='left')
    return feature_set.fillna(0)

def get_today_purchased(date):
    todayData = newdata[(newdata.date == date)]
    
    results = pd.crosstab([todayData.user_id, todayData.item_id], todayData.behavior_type)
    results.columns = ['ui_view', 'ui_love', 'ui_add', 'ui_buy']
    purchased_today = results[results.ui_buy > 0]['ui_buy'].apply(lambda x:1).reset_index(name='label')
    return purchased_today

def get_train_set(date):
    trainDayBuy = get_today_purchased(date)
    
    # Get history features about yeterday
    feature_set = get_features(date)
    data_set  = pd.merge(feature_set, trainDayBuy, how='left', on=['user_id', 'item_id']).fillna(0)
    return data_set

def trainingData(testDay):
    end_date = testDay - datetime.timedelta(1)
    endDay = end_date.strftime('%Y-%m-%d')
    train_dates = pd.date_range('2014-12-02', endDay, freq='D').date
    dataSets = pd.DataFrame()

    # Concate daily feature
    for day in train_dates:
        train_set = get_train_set( day)
        dataSets = dataSets.append(train_set)
        dataSets.drop_duplicates()
    return dataSets


'''
import produceDataSet
import datetime

#import imp
#imp.reload(produceDataSet)

test_date = datetime.date(2014, 12, 4)
testDay = test_date.strftime('%Y-%m-%d')

test_set = produceDataSet.get_features(test_date)
testDay_purchased = produceDataSet.get_today_purchased(test_date)
trainSets = produceDataSet.trainingData(test_date)

train_sets.to_csv('train_sets'+testDay+'.csv')
test_set.to_csv('test_set'+testDay+'.csv')
testDay_purchased.to_csv('purcahsedOntestDay'+testDay+'.csv')
'''