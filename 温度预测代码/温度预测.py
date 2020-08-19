#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
import statsmodels.tsa.stattools as ts
import tqdm


# # 数据读取

# In[2]:


# def parse(x):
#     return datetime.datetime.strptime(x,'%Y %m %d %H %M %S')

train_df=pd.read_csv('C:/Users/zhouwei/Desktop/温室温度预测挑战赛_温室温度预测赛初赛数据/train/train.csv')
test_df=pd.read_csv('C:/Users/zhouwei/Desktop/温室温度预测挑战赛_温室温度预测赛初赛数据/test/test.csv')
train_df


# # 数据探索

# In[3]:


train_df.info()
test_df.info()


# In[4]:


train_df.columns=['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo', 'temperature']
test_df.columns=['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']


# In[5]:


train_df['date']=pd.to_datetime(train_df['year'].astype(str)+'/'+train_df['month'].astype(str)+'/'+train_df['day'].astype(str)+'/'+train_df['hour'].astype(str)+':'+train_df['min'].astype(str))
test_df['date']=pd.to_datetime(test_df['year'].astype(str)+'/'+test_df['month'].astype(str)+'/'+test_df['day'].astype(str)+'/'+test_df['hour'].astype(str)+':'+test_df['min'].astype(str))


# In[6]:


#发现数据集中时间戳有缺失，所以进行时间戳的补齐
count_df=train_df.groupby('date')['temperature'].count()
mutil_date=count_df.loc[count_df>1].index.tolist()
mean_df=pd.DataFrame()
for ind in np.arange(len(mutil_date)):
    slice_df=train_df.loc[train_df['date']==mutil_date[ind]]
    new_df=pd.DataFrame(index=[ind])
    new_df['outdoorTemp']=slice_df['outdoorTemp'].mean()
    new_df['outdoorHum'] = slice_df['outdoorHum'].mean()
    new_df['outdoorAtmo'] = slice_df['outdoorAtmo'].mean()
    new_df['indoorHum'] = slice_df['indoorHum'].mean()
    new_df['indoorAtmo'] = slice_df['indoorAtmo'].mean()
    new_df['temperature'] = slice_df['temperature'].mean()
    mean_df=pd.concat([mean_df,new_df])
temp=train_df.loc[~train_df['date'].isin(mutil_date)]
combine=pd.concat([temp,mean_df])
combine.sort_values('date',inplace=True)


# In[7]:


#进行时间戳补齐
dt2=pd.to_datetime(pd.date_range("2019-03-14 01:00:00","2019-04-03 00:59:00", freq='1min')).tolist()
dt_3=pd.DataFrame(index=dt2).reset_index()
combine2=pd.merge(dt_3,combine,left_on='index',right_on='date',how='left')


# In[8]:


combine2=combine2.set_index('index')


# In[9]:


train_df=combine2.resample('1T').mean()
train_df=train_df[['time','year','month','day','hour','min','sec','outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo', 'temperature']]
train_df['date']=train_df.index.tolist()


# In[10]:



train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['hour'] = train_df['date'].dt.hour
train_df['min'] = train_df['date'].dt.minute
train_df['outdoorTemp'] = train_df['outdoorTemp'].fillna(method='bfill')
train_df['outdoorHum'] = train_df['outdoorHum'].fillna(method='bfill')
train_df['outdoorAtmo'] = train_df['outdoorAtmo'].fillna(method='bfill')
train_df['indoorHum'] = train_df['indoorHum'].fillna(method='bfill')
train_df['indoorAtmo'] = train_df['indoorAtmo'].fillna(method='bfill')
train_df['temperature'] = train_df['temperature'].fillna(method='bfill')
test_df = test_df.fillna(method='bfill')


# # 对训练集进行数据可视化探索
# 

# In[11]:


#查看各个特征之间小时分布趋势
train_df.groupby('hour')['outdoorTemp'].mean().plot()
plt.show()
train_df.groupby('hour')['outdoorHum'].mean().plot()
plt.show()
train_df.groupby('hour')['outdoorAtmo'].mean().plot()
plt.show()
train_df.groupby('hour')['indoorHum'].mean().plot()
plt.show()
train_df.groupby('hour')['indoorAtmo'].mean().plot()


# In[12]:


#从图上可以看出室内温度分布状况基本相似，而室内外温差异常点均出现在下四分位线以下，但距离不是很大，室内外压强有很多的点分布在上四分线以及下四分位线以外，但是有部分值出现离整体较远，是异常值进行删除
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
train_df[['temperature','outdoorTemp','outdoorHum','indoorHum']].boxplot()
plt.show()
plt.figure(figsize=(15,10))
plt.subplot(2,1,2)
train_df[['indoorAtmo','outdoorAtmo']].boxplot()
plt.show()


# In[13]:


for f2 in tqdm.tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']):
    train_df['ago_1hour_'+f2]=train_df[f2].shift(1*60)
for f2 in tqdm.tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']):
    test_df['ago_1hour_'+f2]=test_df[f2].shift(1*2)


# In[14]:


last_hour=train_df.resample('1h').median().iloc[-1]
for f2 in tqdm.tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']):
    ind = test_df.columns.tolist().index('ago_1hour_'+f2)
    test_df.iloc[0,ind]=last_hour[f2]
    test_df.iloc[1,ind] = last_hour[f2]


# In[15]:


for f3 in tqdm.tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']):
    train_df['ago_1hour_'+f3+'trend ']=train_df[f3]-train_df['ago_1hour_'+f3]
for f3 in tqdm.tqdm(['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']):
    test_df['ago_1hour_'+f3+'trend ']=test_df[f3]-test_df['ago_1hour_'+f3]


# In[16]:


def get_statis_n_period_num(used_data,col,n,n_unit=24):
    data = used_data.copy()
    temp = pd.DataFrame()
    for i in range(n):
        temp = pd.concat([temp,data[col].shift((i+1)*n_unit)],axis=1)
    data['avg'+'_'+str(n)+'_period_'+col] = temp.mean(axis=1)
    data['median'+'_'+str(n)+'_period_'+col] = temp.median(axis=1)
    data['max'+'_'+str(n)+'_period_'+col] = temp.max(axis=1)
    data['min'+'_'+str(n)+'_period_'+col] = temp.min(axis=1)
    data['mad'+'_'+str(n)+'_period_'+col] = temp.mad(axis=1)
    data['std'+'_'+str(n)+'_period_'+col] = temp.std(axis=1)
    data['skew'+'_'+str(n)+'_period_'+col] = temp.skew(axis=1)
    data['kurt'+'_'+str(n)+'_period_'+col] = temp.kurt(axis=1)
    data['q1'+'_'+str(n)+'_period_'+col] = temp.quantile(q=0.25,axis=1)
    data['q3'+'_'+str(n)+'_period_'+col] = temp.quantile(q=0.75,axis=1)
    data['var'+'_'+str(n)+'_period_'+col] = data['std'+'_'+str(n)+'_period_'+col]/data['avg'+'_'+str(n)+'_period_'+col] 
    return data


# In[17]:


train_df = get_statis_n_period_num(train_df,'outdoorTemp',24,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorTemp',15,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorTemp',5,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorHum',24,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorHum',15,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorHum',5,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorAtmo',24,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorAtmo',15,n_unit=60)
train_df = get_statis_n_period_num(train_df,'outdoorAtmo',5,n_unit=60)
train_df = get_statis_n_period_num(train_df,'indoorHum',24,n_unit=60)
train_df = get_statis_n_period_num(train_df,'indoorHum',15,n_unit=60)
train_df = get_statis_n_period_num(train_df,'indoorHum',5,n_unit=60)
train_df = get_statis_n_period_num(train_df,'indoorAtmo',24,n_unit=60)
train_df = get_statis_n_period_num(train_df,'indoorAtmo',15,n_unit=60)
train_df = get_statis_n_period_num(train_df,'indoorAtmo',5,n_unit=60)


# In[18]:


test_df = get_statis_n_period_num(test_df,'outdoorTemp',24,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorTemp',15,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorTemp',5,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorHum',24,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorHum',15,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorHum',5,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorAtmo',24,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorAtmo',15,n_unit=2)
test_df = get_statis_n_period_num(test_df,'outdoorAtmo',5,n_unit=2)
test_df = get_statis_n_period_num(test_df,'indoorHum',24,n_unit=2)
test_df = get_statis_n_period_num(test_df,'indoorHum',15,n_unit=2)
test_df = get_statis_n_period_num(test_df,'indoorHum',5,n_unit=2)
test_df = get_statis_n_period_num(test_df,'indoorAtmo',24,n_unit=2)
test_df = get_statis_n_period_num(test_df,'indoorAtmo',15,n_unit=2)
test_df = get_statis_n_period_num(test_df,'indoorAtmo',5,n_unit=2)


# In[19]:


train_df=train_df.fillna(method='bfill')
test_df=test_df.fillna(method='bfill')


# In[20]:


#对压强去除异常值
Q1_outdoorAtmo=train_df['outdoorAtmo'].quantile(0.25)
Q3_outdoorAtmo=train_df.outdoorAtmo.quantile(0.75)
value1=Q1_outdoorAtmo-1.5*(Q3_outdoorAtmo-Q1_outdoorAtmo)
value2=Q3_outdoorAtmo+1.5*(Q3_outdoorAtmo-Q1_outdoorAtmo)
train_df=train_df.loc[(train_df.outdoorAtmo>value1)&(train_df.outdoorAtmo<value2)]
Q1_indoorAtmo=train_df['indoorAtmo'].quantile(0.25)
Q3_indoorAtmo=train_df.indoorAtmo.quantile(0.75)
value1=Q1_indoorAtmo-1.5*(Q3_indoorAtmo-Q1_indoorAtmo)
value2=Q3_indoorAtmo+1.5*(Q3_indoorAtmo-Q1_indoorAtmo)
train_df=train_df.loc[(train_df.indoorAtmo>value1)&(train_df.indoorAtmo<value2)]

train_df[['indoorAtmo','outdoorAtmo']].boxplot()
plt.show()
train_df[['temperature','outdoorTemp','outdoorHum','indoorHum']].boxplot()


# In[21]:


# 离散化进行特征组合
def divide(data_df):
    for f in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
        data_df[f+'_20_bin'] = pd.cut(data_df[f], 20, duplicates='drop').apply(lambda x:x.left).astype(int)
        data_df[f+'_50_bin'] = pd.cut(data_df[f], 50, duplicates='drop').apply(lambda x:x.left).astype(int)
        data_df[f+'_100_bin'] = pd.cut(data_df[f], 100, duplicates='drop').apply(lambda x:x.left).astype(int)
        data_df[f+'_200_bin'] = pd.cut(data_df[f], 200, duplicates='drop').apply(lambda x:x.left).astype(int)
    return data_df

def combine_20(data_df):
    for f1 in tqdm.tqdm(['outdoorTemp_20_bin','outdoorHum_20_bin','outdoorAtmo_20_bin','indoorHum_20_bin','indoorAtmo_20_bin']):
        for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
            data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
            data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
            data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
            data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
    return data_df

def combine_50(data_df):
    
    for f1 in tqdm.tqdm(['outdoorTemp_50_bin','outdoorHum_50_bin','outdoorAtmo_50_bin','indoorHum_50_bin','indoorAtmo_50_bin']):
        for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
            data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
            data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
            data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
            data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
    return data_df

def combine_100(data_df):
    for f1 in tqdm.tqdm(['outdoorTemp_100_bin','outdoorHum_100_bin','outdoorAtmo_100_bin','indoorHum_100_bin','indoorAtmo_100_bin']):
        for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
            data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
            data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
            data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
            data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
    return data_df

def combine_200(data_df):
    
    for f1 in tqdm.tqdm(['outdoorTemp_200_bin','outdoorHum_200_bin','outdoorAtmo_200_bin','indoorHum_200_bin','indoorAtmo_200_bin']):
        for f2 in ['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']:
            data_df['{}_{}_medi'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('median')
            data_df['{}_{}_mean'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('mean')
            data_df['{}_{}_max'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('max')
            data_df['{}_{}_min'.format(f1,f2)] = data_df.groupby([f1])[f2].transform('min')
    return data_df


# In[22]:


divide(train_df)
combine_20(train_df)
combine_50(train_df)
combine_100(train_df)
combine_200(train_df)


# In[23]:


divide(test_df)
combine_20(test_df)
combine_50(test_df)
combine_100(test_df)
combine_200(test_df)


# In[24]:


#构建基本特征
def time_feature(df): 
    features=['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']
    for i in features:
        df['mean_{}'.format(i)]=df.groupby(['month','day','hour'])[i].transform('mean')
        df['median_{}'.format(i)]=df.groupby(['month','day','hour'])[i].transform('median')
        df['max_{}'.format(i)]=df.groupby(['month','day','hour'])[i].transform('max')
        df['min_{}'.format(i)]=df.groupby(['month','day','hour'])[i].transform('min')
        df['std_{}'.format(i)]=df.groupby(['month','day','hour'])[i].transform('std')
    return df


# In[25]:


time_feature(train_df)
time_feature(test_df)


# In[26]:


train_df=train_df.drop(columns=["time","year","sec",'date','month','day','hour','min'],axis=1)
test=test_df.drop(columns=["time","year","sec",'date','month','day','hour','min'],axis=1)


# In[27]:


k=int(len(train_df)*0.8)

Y=train_df['temperature']-train_df['outdoorTemp']
X=train_df.drop(columns=['temperature'],axis=1)

x_train,x_val,y_train,y_val=X.iloc[:k,:],X.iloc[k:,:],Y[:k],Y[k:]


# In[40]:


xg=XGBRegressor(max_depth=10, #树的最大深度
    n_estimators=2000, #提升迭代的次数，也就是生成多少基模型
    min_child_weight=150, #一个子集的所有观察值的最小权重和
    colsample_bytree=0.8, #列采样率，也就是特征采样率
    subsample=0.7, #构建每棵树对样本的采样率
    learning_rate=0.01,    # eta通过缩减特征的权重使提升计算过程更加保守，防止过拟合
    random_state=2020,#随机数种子
    tree_method='gpu_hist',
    gamma=100,
    reg_lambda=0.6
 

)
xg.fit(x_train,y_train,eval_metric='rmse',eval_set=[(x_val, y_val)], early_stopping_rounds=100,verbose=True,
)


# In[42]:


xg.predict(test)


# In[43]:


final_result=xg.predict(test)+test_df['outdoorTemp'].values


# In[46]:


result=pd.DataFrame(final_result,index=test_df.time,columns=['temperature'])
result.temperature=round(result.temperature,1)


# In[45]:


result.to_csv('C:/Users/zhouwei/Desktop/温室温度预测挑战赛_温室温度预测赛初赛数据/result.csv')


# In[ ]:





# In[ ]:





# In[79]:


#特征构造（考虑白天黑夜）
class feature_creation(object):
    
    #室内和室外压强差特征构造
    def atmo_gap(df):
        df['atmo_gap']=df['outdoorAtmo']-df['indoorAtmo']
        return df
    
    #室内外湿度差
    def hum_gap(df):
        df['hum_gap']=df['outdoorHum']-df['indoorHum']
        return df 

    #进行时间构建特征
    #按照小时构造每个特征的平均值，以及中位值
    def time_feature(df): 
        features=['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']
        for i in features:
            df['mean_{}'.format(i)]=df.groupby(['month','day','hour'])[i].transform('mean')
            df['median_{}'.format(i)]=df.groupby(['month','day','hour'])[i].transform('median')
        return df
    
    #特征之间进行特征构建
    #在一定时间(小时）和温度下，内外压强和湿度的均值，中值
    def time_temp_feature(df):
        features=['outdoorHum','outdoorAtmo','indoorHum','indoorAtmo']
        for i in features:
            df['time_temp_mean_{}'.format(i)]=df.groupby(['month','day','hour','outdoorTemp'])[i].transform('mean')
            df['time_temp_median_{}'.format(i)]=df.groupby(['month','day','hour','outdoorTemp'])[i].transform('median')
        return df 
    
    #在一定时间(小时）和室外压强下，室内压强，室内温度和湿度的均值，中值 
    def time_outatom_feature(df):
        features=['outdoorTemp','outdoorHum','indoorHum','indoorAtmo']
        for i in features:
            df['time_outatom_mean_{}'.format(i)]=df.groupby(['month','day','hour','outdoorAtmo'])[i].transform('mean')
            df['time_outatom_median_{}'.format(i)]=df.groupby(['month','day','hour','outdoorAtmo'])[i].transform('median')
        return df
    
    #在一定时间(小时）和室内压强下，室外压强，室内温度和湿度的均值，中值 
    def time_inatom_feature(df):
        features=['outdoorTemp','outdoorHum','outdoorAtmo','indoorHum']
        for i in features:
            df['time_inatom_mean_{}'.format(i)]=df.groupby(['month','day','hour','indoorAtmo'])[i].transform('mean')
            df['time_inatom_median_{}'.format(i)]=df.groupby(['month','day','hour','indoorAtmo'])[i].transform('median')
        
        return df
    
    #在一定时间(小时）和室内湿度下，室内外压强，室内温度和室外湿度的均值，中值
    def time_inhum_feature(df):
        features=['outdoorTemp','outdoorAtmo','outdoorHum','indoorAtmo']
        for i in features:
            df['time_inhum_mean_{}'.format(i)]=df.groupby(['month','day','hour','indoorHum'])[i].transform('mean')
            df['time_inhum_median_{}'.format(i)]=df.groupby(['month','day','hour','indoorHum'])[i].transform('median')
            
        return df
    
    ##在一定时间(小时）和室外湿度下，室内外压强，室内温度和室内湿度的均值，中值 
    def time_outhum_feature(df):
        features=['outdoorTemp','outdoorAtmo','indoorHum','indoorAtmo']
        for i in features:
            df['time_outhum_mean_{}'.format(i)]=df.groupby(['month','day','hour','outdoorHum'])[i].transform('mean')
            df['time_outhum_median_{}'.format(i)]=df.groupby(['month','day','hour','outdoorHum'])[i].transform('median')
            
        return df
    
    
    #室外的变量对室内变量的影响，因为室外的温度无法由人为来决定，
    #室外两个对室内因素的影响,室外压强与室外温度对室内因素影响
    
    def ato_temp_feature(df):
        features=['indoorHum','indoorAtmo']
        for i in features:
            df['out_inout_{}_mean'.format(i)]=df.groupby(['month','day','hour','outdoorAtmo','outdoorTemp'])[i].transform('mean')
            df['out_inout_{}_median'.format(i)]=df.groupby(['month','day','hour','outdoorAtmo','outdoorTemp'])[i].transform('median')
            
        return df
    
    #室外压强和室外湿度对室内因素的影响
    def ato_hum_feature(df):
        features=['indoorHum','indoorAtmo']
        for i in features:
            df['ato_hum_{}_mean'.format(i)]=df.groupby(['month','day','hour','outdoorAtmo','outdoorHum'])[i].transform('mean')
            df['ato_hum_{}_median'.format(i)]=df.groupby(['month','day','hour','outdoorAtmo','outdoorHum'])[i].transform('median')
            
        return df
    #室外温度和湿度对室内因素的影响
    def hum_temp_feature(df):
        features=['indoorHum','indoorAtmo']
        for i in features:
            df['hum_temp_{}_mean'.format(i)]=df.groupby(['month','day','hour','outdoorTemp','outdoorHum'])[i].transform('mean')
            df['hum_temp_{}_median'.format(i)]=df.groupby(['month','day','hour','outdoorTemp','outdoorHum'])[i].transform('median')
            
        return df
    
    #eg.在一定时间下*小时，室外温度，压强与湿度下，室内湿度与压强的均值，中值（室外三个因素）
    def time_out_in_feature(df):
        features=['indoorHum','indoorAtmo']
        for i in features:
            df['time_out_in_{}_mean'.format(i)]=df.groupby(['month','day','hour','outdoorHum','outdoorTemp','outdoorAtmo'])[i].transform('mean')
            df['time_out_in_{}_median'.format(i)]=df.groupby(['month','day','hour','outdoorHum','outdoorTemp','outdoorAtmo'])[i].transform('median')
           
        return df


# In[145]:


importances=model.feature_importances_
indices =np.argsort(importances)[::-1][:15]
features =tra_x.columns
for f in range(indices.shape[0]):
    print("%2d) %-*s %f" % (f + 1,30, features[f], importances[indices[f]]))


# In[145]:


ne['humdiff']=ne['outdoorHum']-ne['indoorHum']
ne['atmodiff']=ne['outdoorAtmo']-ne['indoorAtmo']
ne['tempdiff']=ne['outdoorTemp']-ne['temperature']
ne
plt.figure()
plt.plot(ne.humdiff)
plt.plot(ne.tempdiff)


# In[70]:


#对特征小时频率进行可视化
def show_hours(date,variable):
    count={}
    for i in range(1,24):
        date1=date+' %02.d' %i
        value=ne[date1][variable].mean()
        count[date1]=value
    df=pd.DataFrame.from_dict(count,orient='index')
    df.plot(kind='bar')
    plt.legend(loc='legend')


# In[73]:


show_hours('2019-03-21','outdoorHum')


# In[94]:


#进行特征选择
feature_s=SelectKBest(f_regression,k=15)
feature_s.fit(X_train,Y_train)
features=feature_s.get_support().tolist()
columns=list(df_features.columns)
selected_feature=[]
for i in range(0,len(features)):
    if features[i]==True:
        selected_feature.append(columns[i])
    i+=1
        
selected_feature


# In[ ]:


#将时间序列转化成有监督学习
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df =pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg =pd.concat(cols, axis=1)
    agg.columns = names
    # 去掉NaN行
    if dropnan:
        agg.dropna(inplace=True)
    return agg
df_values=series_to_supervised(df_value,1,1)
df_values


# In[97]:


#转换成3D格式【样本数，时间步，特征数】
train_x=x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
val_x=x_val.reshape((x_val.shape[0],1,x_val.shape[1]))
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM


# In[103]:


#先做一个最简单的模型
model_ = Sequential()
model_.add(LSTM(output_dim=100,input_shape=(train_x.shape[1], train_x.shape[2])))
model_.add(Dropout(0.5))
model_.add(Dense(1))
model_.compile(optimizer='adam', loss='mse')
result = model_.fit(train_x, y_train, epochs=500, batch_size=50, validation_data=(val_x, y_val), verbose=2, shuffle=False)


# In[104]:


#进行训练的Loss可视化
line1 = result.history['loss']
line2 = result.history['val_loss']
plt.plot(line1, label='train', c='g')
plt.plot(line2, label='test', c='r')
plt.legend(loc='best')
plt.show()


# In[ ]:


x_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1])
y_predict=model_.predict(x_test)


# In[ ]:


result=pd.DataFrame(y_predict,index=test_df.index,columns=['temperature'])
result.to_csv('C:/Users/zhouwei/Desktop/温室温度预测挑战赛_温室温度预测赛初赛数据/result.csv')

