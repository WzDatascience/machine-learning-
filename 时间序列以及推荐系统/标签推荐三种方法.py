#!/usr/bin/env python
# coding: utf-8

# In[92]:


import random 
import math
import operator
import pandas as pd
df=pd.read_csv('C:/Users/zhouwei/Desktop/code/delicious-2k/user_taggedbookmarks-timestamps.dat',sep='\t')
df.head()


# In[93]:


record={}
for i in range(len(df)):
    userid=df['userID'][i]
    itemid=df['bookmarkID'][i]
    tagid=df['tagID'][i]
    record.setdefault(userid,{})
    record[userid].setdefault(itemid,[])
    record[userid][itemid].append(tagid)


# In[94]:


#划分训练集与测试集
ratio=0.2
test_data={}
train_data={}
for u in record.keys():
    for i in record[u].keys():
        if random.random()<0.2:
            test_data.setdefault(u,{})
            test_data[u].setdefault(i,[])
            for t in record[u][i]:
                test_data[u][i].append(t)
        else:
            train_data.setdefault(u,{})
            train_data[u].setdefault(i,[])
            for t in record[u][i]:
                train_data[u][i].append(t)


# In[95]:


def addValueToMat(mat, index, item):
    if index not in mat:
        mat.setdefault(index,{})
        mat[index].setdefault(item,1)
    else:
        if item not in mat[index].keys():
            mat[index][item] = 1
        else:
            mat[index][item] += 1
    return mat


# In[96]:


#得到user-tags,tage_items,item_tags,user_tags
user_tags=dict()
tag_items=dict()
user_items=dict()
tag_users=dict()
for index,item in train_data.items():
    for key,tags in item.items():
        for tag in tags: 
            addValueToMat(user_tags,index,tag)
            addValueToMat(tag_items,tag,key)
            addValueToMat(user_items,index,key)
            addValueToMat(tag_users,tag,index)


# In[97]:


#将三种方法封装在一个类里面
class recommed:
    #simpletagbased,给不同的用户推荐topN的商品
    recommeditem={}
    def simple(user,top):
        for tag,value in user_tags[user].items():
            if tag in tag_items:
                for item,num in tag_items[tag].items():
                    if item in recommeditem:
                        recommeditem[item]+=value*num
                    else:
                        recommeditem[item]=value*num
        return sorted(recommeditem.items(),key=lambda x:x[1],reverse=True)[:top]
    #normtagbased方法
    recommenditem={}
    def norm(user,top):
        for tag,value in user_items[user].items():
            if tag in tag_items:
                for item,num in tag_items[tag].items():
                    if item in recommenditem:
                        recommenditem[item]+=(value*num)/(len(user_tags[user])*len(tag_items[tag]))
                    else:
                        recommenditem[item]=(value*num)/(len(user_tags[user])*len(tag_items[tag]))
        return sorted(recommenditem.items(),lambda x:x[1],reverse=True)[:top]
    #tftagbased,给用户推荐topN的商品
    recomitem={}
    def tf(user,top):
        for tag,value in user_items[user].items():
            if tag in tag_items:
                for item,num in tag_items[tag].items():
                    if item in recommitem:
                        recommitem[item]+=(value*num)/math.log(1+len(tag_users[tag]))
                    else:
                        recommitem[item]=(value*num)/math.log(1+len(tag_users[tag]))
        return sorted(recommitem.items(),lambda x:x[1],reverse=True)[:top]


# In[114]:


def precisionandrecall(N):
    hit=0
    for user,items in test_data.items():
        if user in train_data:
            #此处可以修改推荐方法
            rank=recommed.simple(user,N)
            for item,_ in rank:
                if item in items:
                    hit=hit+1
            h_recall=len(items)
            h_preciaion=N
    return (hit/h_preciaion),(hit/h_recall)


# In[115]:


def testRecommend(n):
    precision,recall = precisionandrecall(n)
    print('n:{},\n,precision:{},\n,recall:{}'.format(n, precision, recall))


# In[116]:


testRecommend(10)


# In[ ]:




