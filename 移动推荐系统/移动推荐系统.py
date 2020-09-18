#!/usr/bin/env python
# coding: utf-8

# In[4]:


import time, datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import gc
from sklearn.metrics import f1_score1,roc_auc_score,roc_curve
data=pd.read_csv('C:/Users/zhouwei/Desktop/fresh_comp_offline/tianchi_fresh_comp_train_user.csv')


# In[2]:


data.head()


# In[5]:


#将用户行为为3,4的数据提取出来，进行探索
df_34=data[data['behavior_type'].isin([3,4])][['user_id','item_id','behavior_type','time']]
df_34=df_34.drop_duplicates(['user_id','item_id','behavior_type'])
#behavior_type=3的数据
df_3=df_34[df_34.behavior_type==3][['user_id','item_id','time']]
#behavior_type=4d的数据
df_4=df_34[df_34.behavior_type==4][['user_id','item_id','time']]
#回收内存
del df_34,data
gc.collect()
df_3


# In[65]:


#将数据进行连接
df_time=pd.merge(df_3,df_4,on=['user_id','item_id'],how='outer')
df_time34=df_time.dropna()
#创建新的列，用户在加入购物车到购买的时间的间隔
df_time34['time_gap']=(pd.to_datetime(df_time34['time_y'])-pd.to_datetime(df_time34['time_x']))
time_gap=df_time34.time_gap.to_list()
time_gap


# In[76]:


delta_hour=[]
for i in range(len(time_gap)):
    hours=time_gap[i].days*24+time_gap[i].seconds/3600
    if hours<0:
        continue
    else:
        delta_hour.append(hours)
#对数据进行画图
plt.hist(delta_hour,bins=30)
plt.xlabel('hours')
plt.ylabel('count')
plt.title('hours from adding to purchase')

oneday=[]
for i in delta_hour:
    if i<=24:
        oneday.append(i)
print('在一天购买的比例:{}'.format(len(oneday)/len(delta_hour)))


# In[ ]:


# # 从上述可以看出，大部分的人都是在加入购物车后一天内进行购买,约88%的人
# 根据18号的数据对19号进行预测，对那些已经在18号将物品加入到购物车的人假定这些人都会进行在19号进行购买
temp=df_merge['2014-11-18']
temp[temp['behavior_type']==3]


# In[ ]:


pd.options.display.max_columns=100
count_user=df['behavior_type'].value_counts()
#计算CVR购买转换率
count_all=count_user[1]+count_user[2]+count_user[3]+count_user[4]
count_4=count_user[4]
CVR=count_4/count_all
#将time类型设置成datetime 类
data['time']=pd.to_datetime(data['time'])

def get_weekday(i):
    return i.weekday()

data['weekday']=data['time'].apply(get_weekday)+1
data['hour']=data['time'].dt.hour


# In[ ]:


#购买行为数目和浏览数目有很大的差别，后期可以考虑进行样本采样,因为很多人在同一时间进行多次的浏览操作
data['behavior_type'].value_counts()


# In[ ]:


#对所有用户和商品的数据进行小时，以及一周中每天的购买情况进行可视化,从图表上来看，用户从早上的7点左右开始行为开始变得增加，在晚上8点以后到次日凌晨零点比较频繁
df_hour=data.groupby('hour')['behavior_type'].value_counts().unstack()
df_hour.plot(kind='bar',figsize=(10,10))


# In[ ]:


#对一周的行为进行可视化,得出用户在周五时购买行为会比较频繁
plt.rcParams['font.sans-serif'] = ['SimHei']
data[data['behavior_type']==1].groupby('weekday')['behavior_type'].count().plot(kind='bar',title='浏览量')
plt.show()
data[data['behavior_type']==2].groupby('weekday')['behavior_type'].count().plot(kind='bar',title='点击量')
plt.show()
data[data['behavior_type']==3].groupby('weekday')['behavior_type'].count().plot(kind='bar',title='收藏')
plt.show()
data[data['behavior_type']==4].groupby('weekday')['behavior_type'].count().plot(kind='bar',title='购买')


# In[ ]:


df_copy=data.copy().set_index('time')
#对每天的行为数量进行可视化
str='2014-11-18'
temp=datetime.datetime.strptime(str,'%Y-%m-%d')
delta=datetime.timedelta(days=1)
count_day=defaultdict(int)
for _ in range(30):
    temp+=delta
    temp_str=temp.strftime('%Y-%m-%d')
    #统计每天的行为数
    count_day[temp_str]=count_day[temp_str]+df_copy[temp_str].shape[0]   
    
df_count_day=pd.DataFrame.from_dict(count_day,orient='index',columns=['count'])


# In[ ]:


#可以发现用户在双12和12月14日进行的操作比较频繁
df_count_day.plot(kind='bar')
plt.legend(loc='best')
plt.grid(True)


# In[ ]:


#查看对预估对商品子集12月19号之前行为
df_p=pd.read_csv('C:/Users/zhouwei/Desktop/fresh_comp_offline/tianchi_fresh_comp_train_item.csv')
df_merge=pd.merge(df_p,df.reset_index(),on='item_id').set_index('time')
df_merge


# In[ ]:


#对商品子集进行一天所有小时的行为进行可视化。
df_copy=df.copy().set_index('time')
def show_count_hour(date1):
    count_hour={}
    for i in range(24):
        time_str=date1+ ' %02.d'%i
        count_hour[time_str]=[0,0,0,0]
        temp=df_copy[time_str]['behavior_type'].value_counts()
        for j in range(len(temp)):
            count_hour[time_str][temp.index[j]-1]+=temp[temp.index[j]]
    df_count_hour=pd.DataFrame.from_dict(count_hour,orient='index')
    df_count_hour.plot(kind='bar')
    plt.legend(loc='legend')
show_count_hour('2014-12-18')


# In[ ]:


df_merge.drop(columns=['item_geohash','user_geohash'],axis=0,inplace=True)
df_merge=df_merge.drop('item_category_y',axis=1).rename(columns={'item_category_x':'item_category'})



# 将数据分成几个周期进行训练，part1用来做训练，part2用来做验证集，part3用来预测
# 
#     part1：11.22-11.27 > 11.28;
#     part2：11.29-12.04 > 12.05;
#     part3：12.13-12.18 > 12.19;

# In[ ]:


data.time=pd.to_datetime(data.time)
data.set_index('time',inplace=True)
trainset=data['2014-11-22':'2014-11-28']
validationset=data['2014-11-29':'2014-12-05']
testset=data['2014-12-13':'2014-12-19']


# In[ ]:


#添加label,购买的行为1，其它行为为0，成为二分类问题
def labelcreate(data):
    purchase=data[data['behavior_type']==4]
    purchase['label']=1
    others=data[data['behavior_type']!=4]
    others['label']=0
    new_data=pd.concat([purchase,others])
    return new_data


# In[ ]:


#将trainset子集划分训练集和测试集
train_part=trainset['2014-11-22':'2014-11-27']
test_part=trainset['2014-11-28']
test_part=labelcreate(test_part)


# In[ ]:


train_part.reset_index(inplace=True)


# In[ ]:


#将user和用户行为进行计数得出用户行为总数
def u_b_count(data,n,date):
    data=data[data['time'] >= np.datetime64(date)]
    data['cumcount']=data.groupby(['user_id','behavior_type']).cumcount()
    u_b_count=data.drop_duplicates(['user_id','behavior_type'],keep='last')[['user_id','behavior_type','cumcount']]
    u_b_count=pd.get_dummies(u_b_count['behavior_type']).join(u_b_count[['user_id','cumcount']])
    u_b_count.rename(columns={1:'behavior_type1',2:'behavior_type2',3:'behavior_type3',4:'behavior_type4'},inplace=True)
    u_b_count['u_b1_count_in_{}'.format(n)]=u_b_count['behavior_type1']*(u_b_count['cumcount']+1)
    u_b_count['u_b2_count_in_{}'.format(n)]=u_b_count['behavior_type2']*(u_b_count['cumcount']+1)
    u_b_count['u_b3_count_in_{}'.format(n)]=u_b_count['behavior_type3']*(u_b_count['cumcount']+1)
    u_b_count['u_b4_count_in_{}'.format(n)]=u_b_count['behavior_type4']*(u_b_count['cumcount']+1)
    u_b_count=u_b_count.groupby('user_id').agg({'u_b1_count_in_{}'.format(n): np.sum,
                                                          'u_b2_count_in_{}'.format(n): np.sum,
                                                           'u_b3_count_in_{}'.format(n): np.sum,
                                                           'u_b4_count_in_{}'.format(n): np.sum})
    u_b_count['u_b_count_in_{}'.format(n)]=u_b_count[['u_b1_count_in_{}'.format(n),'u_b2_count_in_{}'.format(n),'u_b3_count_in_{}'.format(n), 'u_b4_count_in_{}'.format(n)]].apply(lambda x: x.sum(), axis = 1)
    return u_b_count


# In[ ]:


u_b_count_in_6=u_b_count(train_part,6,'2014-11-22')
u_b_count_in_3=u_b_count(train_part,3,'2014-11-25')
u_b_count_in_1=u_b_count(train_part,1,'2014-11-27')


# In[ ]:


u_b_count_in_1


# In[ ]:


u_b_count_in_6=u_b_count_in_6.reset_index()
u_b_count_in_3=u_b_count_in_3.reset_index()
u_b_count_in_1=u_b_count_in_1.reset_index()


# In[ ]:


merged_df=pd.merge(u_b_count_in_6,u_b_count_in_3,on = ['user_id'],how = 'left')
merged_df=pd.merge(merged_df,u_b_count_in_1,on=['user_id'],how='left')


# In[ ]:


#离观测天前6天的点击购买率
merged_df['u_b4_rate']=merged_df['u_b4_count_in_6']/merged_df['u_b_count_in_6']


# In[ ]:


#用户第一次购买与用户第一次浏览的时间间隔
def time_gap(data):
    data.sort_values(by=['user_id','time'])
    user_first_purchase=data[data['behavior_type']==4].drop_duplicates(['user_id'],'first')[['user_id','time']]
    user_first_purchase.columns=['user_id','b4_first_time']
    #用户第一次行为时间
    user_first_behavior=data.drop_duplicates(['user_id'],keep='first')[['user_id','time']]
    user_first_behavior.columns=['user_id','b_first_time']
    time_gap=pd.merge(user_first_behavior,user_first_purchase,on=['user_id'])
    time_gap['u_b4_diff_time']=time_gap['b4_first_time']-time_gap['b_first_time']
    time_gap['u_b4_diff_hours']=time_gap['u_b4_diff_time'].apply(lambda x:x.days*24+x.seconds//3600)
    return time_gap


# In[ ]:


time_gap_pb=time_gap(train_part)


# In[ ]:


temp1=pd.merge(merged_df,time_gap_pb,on=['user_id'],how='left')[['user_id',
                                                       'u_b1_count_in_6', 
                                                       'u_b2_count_in_6', 
                                                       'u_b3_count_in_6', 
                                                       'u_b4_count_in_6', 
                                                       'u_b_count_in_6',
                                                       'u_b1_count_in_3',
                                                       'u_b2_count_in_3', 
                                                       'u_b3_count_in_3',
                                                       'u_b4_count_in_3', 
                                                       'u_b_count_in_3',
                                                       'u_b1_count_in_1',
                                                       'u_b2_count_in_1', 
                                                       'u_b3_count_in_1',
                                                       'u_b4_count_in_1', 
                                                       'u_b_count_in_1', 
                                                       'u_b4_rate', 
                                                       'u_b4_diff_hours']]


# In[ ]:


temp1


# In[ ]:


#计算商品在testday前n天用户计数
def i_u_count(data,n,date):
    data=data[data['time'] >= np.datetime64(date)]
    i_u_count=data.drop_duplicates(['item_id','user_id'])
    i_u_count['i_u_count_in_{}'.format(n)]=i_u_count.groupby('item_id').cumcount()+1
    i_u_count=i_u_count.drop_duplicates(['item_id'],'last')[['item_id','i_u_count_in_{}'.format(n)]]
    return i_u_count
i_u_count_in1=i_u_count(train_part,1,'2014-11-27')
i_u_count_in3=i_u_count(train_part,3,'2014-11-25')
i_u_count_in6=i_u_count(train_part,6,'2014-11-22')


# In[ ]:


merged_df1=pd.merge(i_u_count_in1,i_u_count_in3,on=['item_id'],how='left')
merged_df1=pd.merge(merged_df1,i_u_count_in6,on=['item_id'],how='left')


# In[ ]:


#计算商品前n天每种行为的数量以及行为总数
def i_b_count(data,n,date):
    data=data[data['time'] >= np.datetime64(date)]
    i_b_count=data.copy()
    i_b_count['cumcount']=i_b_count.groupby(['item_id','behavior_type']).cumcount()+1
    i_b_count=i_b_count.drop_duplicates(['item_id','behavior_type'],'last')[['item_id','behavior_type','cumcount']]
    i_b_count=pd.get_dummies(i_b_count['behavior_type']).join(i_b_count[['item_id','cumcount']])
    i_b_count.rename(columns={1:'behavior_type1',2:'behavior_type2',3:'behavior_type3',4:'behavior_type4'},inplace=True)
    i_b_count['i_b1_count_in{}'.format(n)]=i_b_count['behavior_type1']*i_b_count['cumcount']
    i_b_count['i_b2_count_in{}'.format(n)]=i_b_count['behavior_type2']*i_b_count['cumcount']
    i_b_count['i_b3_count_in{}'.format(n)]=i_b_count['behavior_type3']*i_b_count['cumcount']
    i_b_count['i_b4_count_in{}'.format(n)]=i_b_count['behavior_type4']*i_b_count['cumcount']
    i_b_count=i_b_count[['item_id','i_b1_count_in{}'.format(n),'i_b2_count_in{}'.format(n),'i_b3_count_in{}'.format(n),'i_b4_count_in{}'.format(n)]]
    i_b_count=i_b_count.groupby('item_id').agg({'i_b1_count_in{}'.format(n): np.sum,
                                  'i_b2_count_in{}'.format(n): np.sum,
                                  'i_b3_count_in{}'.format(n): np.sum,
                                  'i_b4_count_in{}'.format(n): np.sum})
    i_b_count.reset_index()
    #计算商品的总行为数
    i_b_count['i_b_count_in{}'.format(n)]=i_b_count['i_b1_count_in{}'.format(n)]+i_b_count['i_b2_count_in{}'.format(n)]+i_b_count['i_b3_count_in{}'.format(n)]+i_b_count['i_b4_count_in{}'.format(n)]
    return i_b_count


# In[ ]:


i_b_count_1=i_b_count(train_part,1,'2014-11-27').reset_index()
i_b_count_3=i_b_count(train_part,3,'2014-11-25').reset_index()
i_b_count_6=i_b_count(train_part,6,'2014-11-22').reset_index()


# In[ ]:


temp_ib=pd.merge(i_b_count_1,i_b_count_3,on='item_id',how='left')
temp_ib=pd.merge(temp_ib,i_b_count_6,on='item_id',how='left')


# In[ ]:


#商品购买率
temp_ib['i_b4_rate']=temp_ib['i_b4_count_in6']/temp_ib['i_b_count_in6']


# In[ ]:


#商品第一次行为与被购买的行为的时间cha
def time_gapib(data):
    
    i_first_purchase=data[data['behavior_type']==4].drop_duplicates(['item_id'],'first')[['item_id','time']]
    i_first_purchase.columns=['item_id','b4_first_time']
    i_first_behavior=data.drop_duplicates(['item_id'],'first')[['item_id','time']]
    i_first_behavior.columns=['item_id','b_first_time']
    i_time=pd.merge(i_first_behavior,i_first_purchase,on='item_id',how='left')
    i_time['time_diff']=i_time['b4_first_time']-i_time['b_first_time']
    i_time['i_b4_diff_hours']=i_time['time_diff'].apply(lambda x: x.days*24+x.seconds//3600)
    i_time=i_time[['item_id','i_b4_diff_hours']]
    return i_time


# In[ ]:


time_gapib=time_gapib(train_part)


# In[ ]:


time_gapib


# In[ ]:


#将所有与item有关的特征合并
temp2=pd.merge(merged_df1,temp_ib,on='item_id',how='left')
temp2=pd.merge(temp2,time_gapib,on='item_id',how='left')


# In[ ]:


temp2


# In[ ]:


#进行类别与用户之间关系特征构造
def c_u_count(data,n,date):
    data=data[data['time'] >= np.datetime64(date)]
    c_u_count=data.drop_duplicates(['item_category','user_id'])
    c_u_count['c_u_count_in_{}'.format(n)]=c_u_count.groupby('item_category').cumcount()+1
    c_u_count=c_u_count.drop_duplicates(['item_category'],'last')[['item_category','c_u_count_in_{}'.format(n)]]
    return c_u_count
c_u_count_in1=c_u_count(train_part,1,'2014-11-27')
c_u_count_in3=c_u_count(train_part,3,'2014-11-25')
c_u_count_in6=c_u_count(train_part,6,'2014-11-22')
c_u_count_in1


# In[ ]:


merged_cu=pd.merge(c_u_count_in1,c_u_count_in3,on=['item_category'],how='left')
merged_cu=pd.merge(merged_cu,c_u_count_in6,on=['item_category'],how='left')


# In[ ]:


#每个商品类别与每个行为之间的数量
def c_b_count(data,n,date):
    c_b_count=data[data['time'] >= np.datetime64(date)]
    
    c_b_count['cumcount']=c_b_count.groupby(['item_category','behavior_type']).cumcount()+1
    c_b_count=c_b_count.drop_duplicates(['item_category','behavior_type'],'last')[['item_category','behavior_type','cumcount']]
    c_b_count=pd.get_dummies(c_b_count['behavior_type']).join(c_b_count[['item_category','cumcount']])
    c_b_count.rename(columns={1:'behavior_type1',2:'behavior_type2',3:'behavior_type3',4:'behavior_type4'},inplace=True)
    c_b_count['c_b1_count_in{}'.format(n)]=c_b_count['behavior_type1']*c_b_count['cumcount']
    c_b_count['c_b2_count_in{}'.format(n)]=c_b_count['behavior_type2']*c_b_count['cumcount']
    c_b_count['c_b3_count_in{}'.format(n)]=c_b_count['behavior_type3']*c_b_count['cumcount']
    c_b_count['c_b4_count_in{}'.format(n)]=c_b_count['behavior_type4']*c_b_count['cumcount']
    c_b_count=c_b_count[['item_category','c_b1_count_in{}'.format(n),'c_b2_count_in{}'.format(n),'c_b3_count_in{}'.format(n),'c_b4_count_in{}'.format(n)]]
    c_b_count=c_b_count.groupby('item_category').agg({'c_b1_count_in{}'.format(n): np.sum,
                                  'c_b2_count_in{}'.format(n): np.sum,
                                  'c_b3_count_in{}'.format(n): np.sum,
                                  'c_b4_count_in{}'.format(n): np.sum})
    
    #计算商品的总行为数
    c_b_count['c_b_count_in{}'.format(n)]=c_b_count['c_b1_count_in{}'.format(n)]+c_b_count['c_b2_count_in{}'.format(n)]+c_b_count['c_b3_count_in{}'.format(n)]+c_b_count['c_b4_count_in{}'.format(n)]
    return c_b_count


# In[ ]:


c_b_count_in1=c_b_count(train_part,1,'2014-11-27').reset_index()
c_b_count_in3=c_b_count(train_part,3,'2014-11-25').reset_index()
c_b_count_in6=c_b_count(train_part,6,'2014-11-22').reset_index()


# In[ ]:


c_b_count_in1


# In[ ]:


merged_cb=pd.merge(c_b_count_in1,c_b_count_in3,on='item_category',how='left')
merged_cb=pd.merge(merged_cb,c_b_count_in6,on='item_category',how='left')


# In[ ]:


merged_cb['c_b4_rate']=merged_cb['c_b4_count_in6']/merged_cb['c_b_count_in6']


# In[ ]:


#商品类别第一次行为与被购买的行为的时间cha
def time_gapcb(data):
    ctimediff=data
    ctimediff.sort_values(by=['item_category','time'])
    c_first_purchase=ctimediff[ctimediff['behavior_type']==4].drop_duplicates(['item_category'],'first')[['item_category','time']]
    c_first_purchase.columns=['item_category','b4_first_time']
    c_first_behavior=ctimediff.drop_duplicates(['item_category'],'first')[['item_category','time']]
    c_first_behavior.columns=['item_category','b_first_time']
    c_time=pd.merge(c_first_behavior,c_first_purchase,on='item_category',how='left')
    c_time['c_time_diff']=c_time['b4_first_time']-c_time['b_first_time']
    c_time['c_b4_diff_hours']=c_time['c_time_diff'].apply(lambda x: x.days*24+x.seconds*3600)
    c_time=c_time[['item_category','c_b4_diff_hours']]
    return c_time


# In[ ]:


time_gapcb=time_gapcb(train_part)


# In[ ]:


temp3=pd.merge(merged_cu,merged_cb,on='item_category',how='left')
temp3=pd.merge(temp3,time_gapcb,on='item_category',how='left')
temp3


# In[ ]:


#计算用户对商品在n天之前的行为数和行为总数
def ui_b_count(data,n,date):
    ui_b_count=data[data['time']>=np.datetime64(date)]
    ui_b_count['cumcount']=ui_b_count.groupby(['user_id','item_id','behavior_type']).cumcount()+1
    ui_b_count=ui_b_count.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','cumcount']]
    ui_b_count=pd.get_dummies(ui_b_count['behavior_type']).join(ui_b_count[['user_id','item_id','cumcount']])
    ui_b_count=ui_b_count.rename(columns={1:'behavior_type1',2:'behavior_type2',3:'behavior_type3',4:'behavior_type4'})
    ui_b_count['ui_b1_count_in{}'.format(n)]=ui_b_count['behavior_type1']*ui_b_count['cumcount']
    ui_b_count['ui_b2_count_in{}'.format(n)]=ui_b_count['behavior_type2']*ui_b_count['cumcount']
    ui_b_count['ui_b3_count_in{}'.format(n)]=ui_b_count['behavior_type3']*ui_b_count['cumcount']
    ui_b_count['ui_b4_count_in{}'.format(n)]=ui_b_count['behavior_type4']*ui_b_count['cumcount']
    ui_b_count=ui_b_count[['user_id','item_id','ui_b1_count_in{}'.format(n),'ui_b2_count_in{}'.format(n),'ui_b3_count_in{}'.format(n),'ui_b4_count_in{}'.format(n)]]
    ui_b_count=ui_b_count.groupby(['user_id', 'item_id']).agg({'ui_b1_count_in{}'.format(n): np.sum,
                                                          'ui_b2_count_in{}'.format(n): np.sum,
                                                           'ui_b3_count_in{}'.format(n): np.sum,
                                                          'ui_b4_count_in{}'.format(n): np.sum})
    ui_b_count=ui_b_count.reset_index()
    ui_b_count['ui_b_count_in{}'.format(n)]=ui_b_count['ui_b1_count_in{}'.format(n)]+ui_b_count['ui_b2_count_in{}'.format(n)]+ui_b_count['ui_b3_count_in{}'.format(n)]+ui_b_count['ui_b4_count_in{}'.format(n)]
    return ui_b_count


# In[ ]:


ui_b_count_in1=ui_b_count(train_part,1,'2014-11-27')
ui_b_count_in3=ui_b_count(train_part,3,'2014-11-25')
ui_b_count_in6=ui_b_count(train_part,6,'2014-11-22')


# In[ ]:


ui_b_count_in1


# In[ ]:


temp4=pd.merge(ui_b_count_in1,ui_b_count_in3,on=['user_id','item_id'],how='left')
temp4=pd.merge(temp4,ui_b_count_in6,on=['user_id','item_id'],how='left')


# In[ ]:


# ui_b_last_time=train_part.sort_values(by=['time'])
# ui_b_last_time=ui_b_last_time.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','time']]
# ui_b_last_time=ui_b_last_time['ui_b1_last_time']=ui_b_last_time[ui_b_last_time['behavior_type']==1]['time']
# ui_b_last_time=ui_b_last_time['ui_b2_last_time']=ui_b_last_time[ui_b_last_time['behavior_type']==2]['time']
# ui_b_last_time=ui_b_last_time['ui_b3_last_time']=ui_b_last_time[ui_b_last_time['behavior_type']==3]['time']
# ui_b_last_time=ui_b_last_time['ui_b4_last_time']=ui_b_last_time[ui_b_last_time['behavior_type']==4]['time']


# In[ ]:


#计算用户对类别在n天之前的行为数和行为总数
def uc_b_count(data,n,date):
    uc_b_count=data[data['time']>=np.datetime64(date)]
    uc_b_count['cumcount']=uc_b_count.groupby(['user_id','item_category','behavior_type']).cumcount()+1
    uc_b_count=uc_b_count.drop_duplicates(['user_id','item_category','behavior_type'],'last')[['user_id','item_category','behavior_type','cumcount']]
    uc_b_count=pd.get_dummies(uc_b_count['behavior_type']).join(uc_b_count[['user_id','item_category','cumcount']])
    uc_b_count=uc_b_count.rename(columns={1:'behavior_type1',2:'behavior_type2',3:'behavior_type3',4:'behavior_type4'})
    uc_b_count['uc_b1_count_in{}'.format(n)]=uc_b_count['behavior_type1']*uc_b_count['cumcount']
    uc_b_count['uc_b2_count_in{}'.format(n)]=uc_b_count['behavior_type2']*uc_b_count['cumcount']
    uc_b_count['uc_b3_count_in{}'.format(n)]=uc_b_count['behavior_type3']*uc_b_count['cumcount']
    uc_b_count['uc_b4_count_in{}'.format(n)]=uc_b_count['behavior_type4']*uc_b_count['cumcount']
    uc_b_count=uc_b_count[['user_id','item_category','uc_b1_count_in{}'.format(n),'uc_b2_count_in{}'.format(n),'uc_b3_count_in{}'.format(n),'uc_b4_count_in{}'.format(n)]]
    uc_b_count=uc_b_count.groupby(['user_id', 'item_category']).agg({'uc_b1_count_in{}'.format(n): np.sum,
                                                          'uc_b2_count_in{}'.format(n): np.sum,
                                                           'uc_b3_count_in{}'.format(n): np.sum,
                                                          'uc_b4_count_in{}'.format(n): np.sum})
    uc_b_count=uc_b_count.reset_index()
    uc_b_count['uc_b_count_in{}'.format(n)]=uc_b_count['uc_b1_count_in{}'.format(n)]+uc_b_count['uc_b2_count_in{}'.format(n)]+uc_b_count['uc_b3_count_in{}'.format(n)]+uc_b_count['uc_b4_count_in{}'.format(n)]
    return uc_b_count


# In[ ]:


uc_b_count_in1=uc_b_count(train_part,1,'2014-11-27')
uc_b_count_in3=uc_b_count(train_part,3,'2014-11-25')
uc_b_count_in6=uc_b_count(train_part,6,'2014-11-22')


# In[ ]:


uc_b_count_in1


# In[ ]:


temp5=pd.merge(uc_b_count_in1,uc_b_count_in3,on=['user_id','item_category'],how='left')
temp5=pd.merge(temp5,uc_b_count_in6,on=['user_id','item_category'],how='left')


# In[ ]:


temp2


# In[ ]:


temp4


# In[ ]:


#将特征集合并在一起,和需要预测的结果
final=pd.merge(temp5,temp1,on='user_id',how='inner')
final=pd.merge(final,temp3,on='item_category',how='inner')
final=pd.merge(final,temp2,on=['item_id'],how='inner')
final=pd.merge(final,temp4,on=['user_id','item_category'],how='inner')
target=test_part[['user_id','item_category','label']]
final=pd.merge(final,target,on=['user_id','item_category'],how='inner')


# In[ ]:


#去掉不需要的列
final=final.drop(columns=['user_geohash','user_id','item_category','time'],axis=1)
X=final.drop(columns=['label'],axis=1)
Y=final['label']


# In[ ]:


#使用Xgboost进行模型训练
xg=XGBRegressor(max_depth=10, 
    n_estimators=500, 
    min_child_weight=150, 
    colsample_bytree=0.8, 
    subsample=0.7,
    learning_rate=0.01,   
    random_state=2020,
    tree_method='gpu_hist',
    gamma=100,
    reg_lambda=0.6

xg.fit(X,Y)


# In[ ]:


#预测结果并使用f1_score进行测试
y_predict=xg.predict(x_test)
f1_score=f1_score(Y,y_predict)


# In[ ]:


y_predict.to_csv('/submission.csv',index=False)


# In[ ]:


#使用GBDT+LR进行模型训练，GBDT自动构造特征,LR用来分类,将数据集一分成两部分，一部分用来GBDT一部分用于LR
xhalf1,xhalf2,yhalf1,yhalf2=train_test_split(X,Y,train_size=0.5,random_state=2020)
clf=GradientBoostingRegressor(n_estimators=50,learning_rate=0.01,max_depth=3,min_samples_split=2,random_state=000)
clf.fit(xhalf1,yhalf1)
onehot=OneHotEncoder(categories='auto')
onehot.fit(clf.apply(x_half1))

lr=LogisticRegression(solver='lbfgs',max_iter=1000)
lr.fit(onehot.transform(clf.apply(xhalf2)),yhalf2)


# In[ ]:


#使用11-29-12-05的数据进行之前的特征工程，然后进行验证结果,使用roc进行结果检测
y_predict=onehot.transform(clf.apply(x_val)))
fp,tp,_=roc_curve(y_val,predict)
#对12、13-12、18号数据进行特征构造，用来预估12.19号的商品购买
y=onehot.transform(clf.apply(x_test))


# In[ ]:


y.to_csv('/submission.csv',index=False)


# In[ ]:




