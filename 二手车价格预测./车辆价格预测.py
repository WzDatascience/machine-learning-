#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[2]:


train=pd.read_csv('C:/Users/zhouwei/Desktop/ershouche/used_car_train_20200313.csv',sep=' ')
test=pd.read_csv('C:/Users/zhouwei/Desktop/ershouche/used_car_testB_20200421.csv',sep=' ')


# In[3]:


pd.set_option('max_columns',100)
train.describe()


# In[4]:


#查看缺失值
train.isnull().sum()


# In[ ]:


#对数据进行探索
train.fuelType.value_counts()
train.bodyType.isnull().value_counts()
train.bodyType.value_counts()
train.gearbox.isnull().value_counts()
train.gearbox.value_counts()
train.notRepairedDamage.value_counts()


# In[ ]:


#对缺失进行画图
msno.matrix(train)
msno.heatmap(train)
msno.bar(train)


# In[ ]:


#对车辆的类型进行可视化，微型车，箱型车以及大巴车占大部分，特征不是正态分布。
train.bodyType.value_counts().plot.bar()
print('偏度:',train.bodyType.skew(),'峰度:',train.bodyType.kurt())
train.fuelType.value_counts().plot.bar()
print(train.fuelType.kurt(),train.fuelType.skew())
train.model.value_counts().plot.hist()
print('偏度:',train.model.skew(),'峰度:',train.model.kurt())


# In[5]:


#对model进行数据填充#从上面的分布可以看出，空缺的特征非正态分布，不使用均值进行填充,使用中位数或众数填充
train.model=train.model.fillna(0)
train.fuelType=train.fuelType.fillna(0)
#对bodyType进行数据填充
train.bodyType=train.bodyType.fillna(0)
#对gearbox进行数据填充
train.gearbox=train.gearbox.fillna(0)
#将- 换成0
train.notRepairedDamage=train.notRepairedDamage.replace('-',0)


# In[6]:


#seller数据相差太大，没有必要进行探索
train.seller.value_counts()
#数据只有一个类型，没有意义进行探索
train.offerType.value_counts()


# In[ ]:


#进行价格可视化,价格不符合正态分布
train.price.value_counts()
y=train['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)


# In[ ]:


#进行价格频数的可视化的，价格大部分小于20000,在进行价格log之后，数据开始服从正态分布
np.log(train.price).plot.hist()


# In[16]:


train=train.drop(columns=['seller','offerType'],axis=1)


# In[ ]:


#对数据进行baseline 建模
X=train.drop(columns=['price'],axis=1)
Y=train['price']
model=XGBRegressor(n_estimators=200,max_depth=10,random_state=2020,learning_rate=0.01)
model.fit(X,Y)
#对数据进行预测
test=test.fillna(-1)
test.notRepairedDamage=test.notRepairedDamage.replace('-',0)
test.notRepairedDamage=test.notRepairedDamage.astype('float')
test=test.drop(columns=['SaleID', 'name', 'regDate'],axis=1)
y_predict=model.predict(test)


# In[ ]:


对数据进行更细的处理和特征工程的构建


# In[17]:


train.describe()


# In[18]:


train[['price','power']].boxplot()


# In[19]:


#对数据异常值进行处理，从上面describe来看，price最小值只有11,最大有99999，power值也相差比较大
def outliers_proc(data,col_name,scale=3):
    def box_plot_outliers(data,box_scaler):
        iq=data.quantile(0.75)-data.quantile(0.25)
        low_level=data.quantile(0.25)-iq
        high_level=data.quantile(0.75)+iq*1.5
        rule_low=(data<low_level)
        rule_up=(data>high_level)
        return (rule_low,rule_up),(low_level,high_level)
    
    data_n=data.copy()
    data_series=data_n[col_name]
    rule,value=box_plot_outliers(data_series,box_scaler=scale)
    index=np.arange(data_series.shape[0])[rule[0]|rule[1]]
    print('需要去掉的个数:',len(index))
    data_n=data_n.drop(index)
    data_n.reset_index(drop=True,inplace=True)
    print('现在的columns 数：',data_n.shape[0])
    index_low=np.arange(data_series.shape[0])[rule[0]]
    outliers=data_series.iloc[index_low]
    print('description of data less than lower bound is:')
    print(pd.Series(outliers).describe())
    index_up=np.arange(data_series.shape[0])[rule[1]]
    outliers=data_series.iloc[index_up]
    print('description of data larger than the upper bound is:')
    print(pd.Series(outliers).describe())
    
    fig,ax=plt.subplots(1,2,figsize=(10,7))
    sns.boxplot(y=data[col_name],data=data,palette='Set1',ax=ax[0])
    sns.boxplot(y=data_n[col_name],data=data,palette='Set1',ax=ax[1])
    
    return data_n
    


# In[20]:


train=outliers_proc(train,'power',scale=3)


# In[21]:


train[['price']].boxplot()


# In[22]:


train=outliers_proc(train,'price',scale=3)


# In[23]:


train.describe()


# In[24]:


#时间维度特征构建,由于有些日期错误，所以使用coerce
import datetime
def stime(x):
    date=pd.to_datetime(x,format='%Y%m%d',errors='coerce')
    return date


# In[25]:


train.regDate=train.regDate.apply(stime)
train.creatDate=train.creatDate.apply(stime)


# In[26]:


#计算出二手汽车使用年数
train['time_used']=(train.creatDate-train.regDate).dt.days/365


# In[27]:


#由于有些时间有问题，所以出现值为0的现象
train.time_used.isnull().sum()


# In[28]:


#对特征进行统计指标计算，对车的model.brand,bodyType进行车的power和行程数进行groupby
def feature_creation(x):
    columns=['model','brand','bodyType']
    for i in columns:
        train['{}_{}_min'.format(i,x)]=train.groupby(i)[x].transform('min')
        train['{}_{}_max'.format(i,x)]=train.groupby(i)[x].transform('max')
        train['{}_{}_mean'.format(i,x)]=train.groupby(i)[x].transform('mean')
        train['{}_{}_median'.format(i,x)]=train.groupby(i)[x].transform('median')
        train['{}_{}_std'.format(i,x)]=train.groupby(i)[x].transform('std')
    return train


# In[29]:


feature_creation('power')
feature_creation('kilometer')


# In[30]:


pd.set_option('max_rows',100)
train.isnull().sum()


# In[32]:


#去掉不需要的列
train=train.drop(columns=['SaleID','name','brand','model','regDate','regionCode','creatDate'],axis=1)


# In[34]:


train.describe()


# In[38]:


X=train.drop(columns=['price'],axis=1)
Y=np.log(train.price)
#对数据进行正则化
model_standard=StandardScaler()
X_train=pd.DataFrame(model_standard.fit_transform(X),columns=X.columns)


# In[39]:


#对样本进行打乱挑选
x_train,x_val,y_train,y_val=train_test_split(X_train,Y,train_size=0.8,random_state=0000)


# In[51]:


#首先使用xgb，lgbm进行简单建模，得到0.12的mae
def xgb(x_train,y_train,x_val,y_val):
    xgb=XGBRegressor(n_estimators=1000,max_depth=10,learning_rate=0.01,subsample=0.8,colsample_bytree=0.8,random_state=2000)
    xgb.fit(x_train,y_train)
    result=xgb.predict(x_val)
    score=mean_absolute_error(result,y_val)
    return score 
def lgb(x_train,y_train,x_val,y_val):
    lgb=LGBMRegressor(n_estimators=1000,max_depth=10,subsample=0.8,colsample_bytree=0.8,learning_rate=0.01,random_state=2020)
    lgb.fit(x_train,y_train)
    result=lgb.predict(x_val)
    score=mean_absolute_error(result,y_val)
    return score
lgb(x_train,y_train,x_val,y_val)


# In[ ]:


xgb(x_train,y_train,x_val,y_val)


# In[ ]:


#由于样本中含有许多的空值，所以使用xgboost,lightgbm进行模型训练,并使用Gridsearch进行参数调优,当然也可以将空值删除使用其他模型
def tree(x_train,y_train,x_val,y_val):
    tree=RandomForestregressor(n_estimators=100,max_depth=10,random_statemean_absolute_error)
    tree.fit(x_train,y_train)
    result=tree.predict(x_val)
    score=mean_absolute_error(result,y_val)
    return score 

def xgb(x_train,y_train,x_val,y_val):
    xgb=XGBregressor(n_estimators=1000,max_depth=10,learning_rate=0.01,subsample=0.8,colsample_bytree=0.8,random_state=2000)
    xgb.fit(x_train,y_train)
    result=xgb.predict(x_val)
    score=f1_score(result,y_val)
    return score 

def lgb(x_train,y_train,x_val,y_val):
    lgb=LGBMRregressor(n_estimators=1000,max_depth=10,subsample=0.7,colsample_bytree=0.7,learning_rate=0.01,random_state=2020)
    lgb.fit(x_train,y_train)
    result=lgb.predict(x_val)
    score=mean_absolute_error(result,y_val)
    return score


# In[ ]:


# 使用Gridsearch进行参数调试，如果模型MAE值能够降低，则使用该模型，如果不行，则将特征工程进行精加工

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




