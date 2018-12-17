#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import time
import os
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.001, #to get better accuracy
        'num_leaves': 29,  # we should let it be smaller than 2^(max_depth). Low value to avoid overfitting
        'max_depth': 10,  # -1 means no limit
        'feature_fraction': 0.8, #80% of parameters randomly durting each ietration 
        'min_data_in_leaf': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1


# In[10]:


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }


# In[18]:


print('loading train data...')
train_df = pd.read_csv("train.csv", skiprows=range(1,144903891), nrows=10000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
test_df = pd.read_csv("test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df=train_df.append(test_df)


# In[19]:


del test_df
gc.collect()


# In[22]:


print('Extracting new features from the click_time...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
gc.collect()

print('grouping by ip-day-hour combination...')
gp_ip_day_hour= train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
train_df = train_df.merge(gp_ip_day_hour, on=['ip','day','hour'], how='left')
gc.collect()

print('grouping by ip-app combination...')
gp_ip_app = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp_ip_app, on=['ip','app'], how='left')
gc.collect()


print('grouping by ip-app-os combination...')
gp_ip_app_os = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp_ip_app_os, on=['ip','app', 'os'], how='left')
gc.collect()


# Adding features with var and mean hour to make meaning assumptions
print('grouping by : ip_day_chl_var_hour')
gp_ip_day_chl_var_hour = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
train_df = train_df.merge(gp_ip_day_chl_var_hour, on=['ip','day','channel'], how='left')
gc.collect()

print('grouping by : ip_app_os_var_hour')
gp_ip_app_os_var_hour = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
train_df = train_df.merge(gp_ip_app_os_var_hour, on=['ip','app', 'os'], how='left')
gc.collect()

print('grouping by : ip_app_channel_var_day')
gp_ip_app_channel_var_day = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
train_df = train_df.merge(gp_ip_app_channel_var_day, on=['ip','app', 'channel'], how='left')
gc.collect()

print('grouping by : ip_app_chl_mean_hour')
gp_ip_app_chl_mean_hour = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
print("merging...")
train_df = train_df.merge(gp_ip_app_chl_mean_hour, on=['ip','app', 'channel'], how='left')
gc.collect()


# In[23]:


#deleting all the grouping variables to make space for the model to run and they are not used anywhere else

del gp_ip_day_hour
del gp_ip_app
del gp_ip_app_os
del gp_ip_day_chl_var_hour
del gp_ip_app_os_var_hour
del gp_ip_app_channel_var_day
del gp_ip_app_chl_mean_hour


# In[24]:


print("vars and data type: ")
train_df.info()
train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')


test_df = train_df[len_train:]
val_df = train_df[(len_train-2500000):len_train]
train_df = train_df[:(len_train-2500000)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target = 'is_attributed'
for col in df.target:
    if np.issubdtype(df[col], np.number):
        df[col].values[:] = 0
        
predictors = ['app','device','os', 'channel', 'hour', 'day', 
              'ip_tcount', 'ip_tchan_count', 'ip_app_count',
              'ip_app_os_count', 'ip_app_os_var',
              'ip_app_channel_var_day','ip_app_channel_mean_hour']
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
start_time = time.time()


params = {
    'learning_rate': 0.15,
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced 
}
bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=500, 
                        categorical_features=categorical)

print('[{}]: model training time'.format(time.time() - start_time))


# In[25]:


train_df.head()


# In[14]:


del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced99.csv',index=False)
print("done...")


# In[ ]:




