#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import gc


# In[2]:


raw=pd.read_csv('C:/Users/zhouwei/Desktop/Baidu推荐系统/train.csv')


# In[3]:


raw.head()


# In[4]:


raw.isnull().sum()


# In[5]:


#大部分用户都没有进行评价
raw.is_customer_rate.value_counts()


# In[6]:


raw.order_detail_status.value_counts()


# In[7]:


raw.goods_status.value_counts()


# In[8]:


#对数据集中u用户进行复购行为时间进行统计。
df=raw[['customer_id','order_pay_time']]
df=df.sort_values(by=['order_pay_time'])
purchase_first_time=df.drop_duplicates('customer_id',keep='first')
purchase_last_time=df.drop_duplicates('customer_id',keep='last')
df_time=pd.merge(purchase_first_time,purchase_last_time,on='customer_id',how='outer')


# In[9]:


df_time.head()
df_time.dropna()


# In[10]:


df_time['time_gap']=(pd.to_datetime(df_time.order_pay_time_y)-pd.to_datetime(df_time.order_pay_time_x))
#得出用户在进行重新购买的月数
df_time.time_gap=df_time.time_gap.dt.days/30


# In[11]:


#time_gap等于0说明用户没有进行重新购买的行为
df_repurchase=df_time[df_time.time_gap>0]
#将复购的时间进行可视化,大部分的用户会在3个月以内进行复购行为
plt.hist(df_repurchase.time_gap)


# In[24]:


#对会员折扣进行数据探索,从数据上显示大约有92%的商品并不支持会员折扣，所以对于大量会员数据#
#缺失问题，可以将会员的特征进行删除，基本上可以假设会员没有影响
raw.groupby('goods_has_discount')['goods_id'].count()[0]/raw.shape[0]


# In[12]:


#对商品id进行标签编码
raw['goods_id']=pd.factorize(raw.goods_id)[0]


# In[13]:


def preprocess(train):
    #对数据进行特征构造(从用户的维度)
    data=pd.DataFrame(raw.groupby('customer_id')['customer_gender'].last().fillna(0))
    #添加新的列（针对每个用户，最后一次的行为）商品
    data[['order_id_last','goods_id_last','goods_status_last','goods_price_last','goods_has_discount_last','goods_list_time_last','goods_delist_time_last']]=raw.groupby('customer_id')['order_id','goods_id','goods_status','goods_price','goods_has_discount','goods_list_time','goods_delist_time'].last()
    #针对订单的最后一次行为添加新的列
    data[['order_total_num_last','order_amount_last','order_total_payment_last','order_total_discount_last','order_pay_time_last','order_status_last','order_count_last','is_customer_rate_last','order_detail_status_last','order_detail_goods_num_last','order_detail_amount_last','order_detail_payment_last']]=raw.groupby('customer_id')['order_total_num','order_amount','order_total_payment','order_total_discount','order_pay_time','order_status','order_count','is_customer_rate','order_detail_status','order_detail_goods_num','order_detail_amount','order_detail_payment'].last()
    #对商品的原始价格进行统计字段
    data[['goods_price_mean','goods_price_min','goods_price_max','goods_price_median']]=raw.groupby('customer_id').agg({'goods_price':['mean','min','max','median']})
    data[['goods_price_std']]=raw.groupby('customer_id')['goods_price'].std(ddof=0)
    #订单实付金额统计字段
    data[['order_detail_payment_mean','order_detail_payment_min','order_detail_payment_max','order_detail_payment_median']]=raw.groupby('customer_id').agg({'order_detail_payment':['mean','min','max','median']})
    data[['order_detail_payment_std']]=raw.groupby('customer_id')['order_detail_payment'].std(ddof=0)
    #用户购买的订单数量vf
    data[['order_count']]=raw.groupby('customer_id')['order_id'].count()
    #用户购买的商品数量
    data[['goods_count']]=raw.groupby('customer_id')['order_detail_goods_num'].sum()
    #用户所在省份
    # data[['province']]=raw(raw['customer_id'].isin(data.index.to_list())['customer_province']
    #用户购买goods数量
    data[['goods_amount']]=raw.groupby('customer_id')['goods_id'].nunique()
    #商品折扣统计属性
    data[['order_total_discount_mean','order_total_discount_min','order_total_discount_max','order_total_discount_median']]=raw.groupby('customer_id').agg({'order_total_discount':['mean','min','max','median']})
    data[['order_total_discount_std']]=raw.groupby('customer_id')['order_total_discount'].std(ddof=0)
    #订单商品数量统计属性
    data[['order_detail_goods_num_mean','order_detail_goods_num_min','order_detail_goods_num_max','order_detail_goods_num_median']]=raw.groupby('customer_id').agg({'order_detail_goods_num':['mean','min','max','median']})
    data[['order_detail_goods_num_std']]=raw.groupby('customer_id')['order_detail_goods_num'].std(ddof=0)
    #商品最新上架时间diff,以‘2013-01-01 00:00:00’为例
    t=datetime.datetime.strptime('2013-01-01 00:00:00','%Y-%m-%d %H:%M:%S')
    data[['goods_list_time_diff']]=data['goods_list_time_last'].map(lambda x:(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')-t).days/365)
    data[['goods_delist_time_diff']]=data['goods_delist_time_last'].map(lambda x:(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')-t).days/365)
    #商品上架与下架时间差
    data[['goods_delist_list_timediff']]=((pd.to_datetime(data['goods_delist_time_last'])-pd.to_datetime(data['goods_list_time_last'])).dt.days)/365
    #商品支付时间与上下架时间差
    data[['goods_payment_list_timediff']]=((pd.to_datetime(data['goods_list_time_last'])-pd.to_datetime(data['order_pay_time_last'])).dt.days)/365
    data[['goods_payment_delist_timediff']]=((pd.to_datetime(data['goods_delist_time_last'])-pd.to_datetime(data['order_pay_time_last'])).dt.days)/365
    
    return data


# In[14]:





# In[16]:


#从订单的维度进行特征构建
def order_feature(data):
    #最后一个订单的订单数量
    data=pd.DataFrame(raw.groupby('order_id')['order_amount'].last())
    #添加新的列（针对每个订单，最后一次的数据）
    data[['order_total_last_payment','order_total_last_discount','order_last_pay_time']]=raw.groupby('order_id')['order_total_payment','order_total_discount','order_pay_time'].last()
    data[['order_last_status','order_last_count','order_last_detail_status','last_order_good_num']]=raw.groupby('order_id')['order_status','order_count','order_detail_status','order_detail_amount'].last()
    data[['order_last_payment','order_last_discount']]=raw.groupby('order_id')['order_detail_payment','order_detail_discount'].last()
    return data

#从商品维度进行构建
def goods_feature(data):
    data=pd.DataFrame(raw.groupby('goods_id')['goods_class_id'].last())
    data[['last_good_status','last_discount','goods_last_list_time','goods_last_detail_time']]=raw.groupby('goods_id')['goods_status','goods_has_discount','goods_list_time','goods_detail_time'].last()
    data[['goods_time_gap']]=((pd.to_datetime(data['goods_last_detail_time'])-pd.to_datetime(data['goods_last_list_time'])).dt.days)/365
    return data 
#还可以从用户商品，用户与订单维度进行构建


# In[14]:


#将8月份之前的数据作为训练集
train=raw[raw['order_pay_time']<='2013-07-31 23:59:59']


# In[ ]:


order_data=order_feature(train).reset_index()
goods_data=goods_feature(train).reset_index()
customer_data=preprocess(train)


# In[16]:


#将用户进行复购的时间作为一个特征
df_time=df_time.set_index('customer_id')
customer_data[['repurchase_month']]=df_time.time_gap


# In[ ]:


#将三个数据集进行合并
merge=pd.merge(customer_data,order_data,how='outer',on='order_id')
merge=pd.merge(merge,goods_data,how='outer',on='goods_id')


# In[17]:


#8月份的数据作为label
label=set(raw[raw['order_pay_time']>'2013-07-31 23:59:59']['customer_id'].dropna())
merge['labels']=merge.index.map(lambda x: int(x in label))


# In[18]:


#将8月份的数据作为test预测9月份的购买
test=raw[raw['order_pay_time']>'2013-07-31 23:59:59']
test=preprocess(test)
test_order_data=order_feature(test).reset_index()
test_goods_data=goods_feature(test).reset_index()
test[['repurchase_month']]=df_time.time_gap
test_merged=pd.merge(test,test_order_data,on='order_id',how='outer')
test_merged=pd.merge(test_merges,test_goods_data,on='goods_id',how='outer')


# In[19]:


merge.isnull().sum()


# In[20]:


test_merged.isnull().sum()


# In[21]:


#将空值进行填充
merged_train=merged.fillna(-1)
merged_test=test_merged.fillna(-1)


# In[22]:


merged_train.columns


# In[23]:


#去掉不需要的特征
final=merged_train.drop(columns=['order_id_last','goods_id_last','goods_list_time_last','goods_delist_time_last', 'order_pay_time_last'],axis=1)
final_test=merged_test.drop(columns=['order_id_last','goods_id_last','goods_list_time_last','goods_delist_time_last', 'order_pay_time_last'],axis=1)


# In[25]:


#对数据进行交叉验证法评估
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import f1_score,make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[26]:


#定义损失函数
def func(y,p):
    loss=sum(y*np.log(p)*40+(1-y)*np.log(1-p))
    return loss


# In[27]:


X=merged_train.drop('labels',axis=1)
Y=merged_train[['labels']]
Y


# In[28]:


#将数据进行归一化
model=MinMaxScaler()
X_train=pd.DataFrame(model.fit_transform(X),columns=X.columns)


# In[29]:


X_train


# In[30]:


result={}
kfold=KFold(n_splits=5,shuffle=True)
for train_set,val_set in kfold.split(X_train):
    x_train,x_val=X_train.iloc[train_set],X_train.iloc[val_set]
    y_train,y_val=Y.iloc[train_set],Y.iloc[val_set]


# In[31]:



#构建jiandan模型函数
def logistic(x_train,y_train,x_val,y_val):
    logistic=LogisticRegression()
    logistic.fit(x_train,y_train)
    result=logistic.predict(x_val)
    score=f1_score(result,y_val)
    return score
def tree(x_train,y_train,x_val,y_val):
    tree=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=2020)
    tree.fit(x_train,y_train)
    result=tree.predict(x_val)
    score=f1_score(result,y_val)
    return score 

def xgb(x_train,y_train,x_val,y_val):
    xgb=XGBClassifier(n_estimators=1000,max_depth=10,learning_rate=0.01,subsample=0.8,colsample_bytree=0.8,random_state=2000)
    xgb.fit(x_train,y_train)
    result=xgb.predict(x_val)
    score=f1_score(result,y_val)
    return score 

def lgb(x_train,y_train,x_val,y_val):
    lgb=LGBMClassifier(n_estimators=1000,max_depth=10,subsample=0.7,colsample_bytree=0.7,learning_rate=0.01,random_state=2020)
    lgb.fit(x_train,y_train)
    result=lgb.predict(x_val)
    score=f1_score(result,y_val)
    return score


# In[ ]:


#进行内存回收
del raw,data
gc.collect()


# In[32]:


logistic(x_train,y_train,x_val,y_val)


# In[33]:


tree(x_train,y_train,x_val,y_val)


# In[ ]:


#进行调参
lg=LGBMRegressor()
xg=XGBRegressor()
n_estimators = [i*100 for i in range(0,20)]
max_depth = [i for i in range(20)]
learning_rate=[i*0.01 for i in range(0,10)]
subsample=[i*0.1 for i in range(0,10)]
colsample_bytree=[i*0.1 for i in range(0,10)]
parameters={'n_estimators':n_estimators,'max_depth':max_depth,'learning_rate':learning_rate,'subsample':subsample,'colsample_bytree':colsample_bytree}
clf=GridSearchCV(xg,parameters,cv=3,scoring='mae')
clf.fit(x_train,y_train)
print('最佳参数：',clf.best_params_)
y_predict=clf.predict(x_val)
mean_absolute_error(y_predict,y_val)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




