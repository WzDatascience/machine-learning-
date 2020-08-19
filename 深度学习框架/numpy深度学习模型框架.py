#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston


# In[71]:


data=load_boston()
x=data.data
y=data.target
len(x)


# In[8]:


from matplotlib import pyplot as plt
plt.scatter(x[:,6],y)


# In[9]:


import random 
def price(x,k,b):
    return k*x+b


# In[44]:


x_rm=x[:,5]
k=random.randint(-100,100)
b=random.randint(-100,100)
random_price=[price(x_rm,k,b) for x_rm in x_rm]


# In[36]:


plt.scatter(x_rm,random_price)


# In[38]:


def loss(y, y_hat):
    return sum((y-y_hat)**2 for y,y_hat in zip(y,y_hat))/len(y)


# In[50]:


try_times=100000
min_loss=float('inf')
best_k,best_d=None,None
for i in range(try_times):
    k=random.random()*200-100
    b=random.random()*200-100
    y_hat=[price(x_m,k,b) for x_m in x_rm]
    current_loss=loss(y,y_hat)
    if current_loss <min_loss:
        min_loss=current_loss
        best_k,best_b=k,b
        print('when time is:{},get_best_k:{},get_best_b:{},min_loss:{}'.format(i,best_k,best_b,min_loss))


# In[61]:


def gradient_k(x,y,y_hat):
    gradient=0
    n=len(y)
    for x,y, y_hat in zip(x,y,y_hat):
        gradient=+(y-y_hat)*x
    return -2*gradient/n
def gradient_b(y,y_hat):
    gradient=0
    n=len(y)
    for y , y_hat in zip(y,y_hat):
        gradient=+(y-y_hat)
    return -2*gradient/n


# In[72]:


import numpy as pd
def loss1(y,y_hat):
    return (abs(y-y_hat) for y, y_hat in zip(y,y_hat))/len(y)
def gradient_k1(x_m):
    gradient=0
    n=len(x_m)
    for x in x_m:
        gradient+=-x
    return gradient/n
def gradient_b1(x_m):
    n=len(x_m)
    return -1/n
    
    


# In[66]:


min_loss=float('inf')
learning_rate=0.1
try_times=1000000
k=random.random()*200-100
b=random.random()*200-100
for i in range(try_times):
    y_hat=[price(x,k,b) for x in x_rm]
    current_loss=loss(y,y_hat)
    if current_loss<min_loss:
        min_loss=current_loss
    k_gradient=gradient_k(x_rm,y,y_hat)
    b_gradient=gradient_b(y,y_hat)
    k=k+(-1*k_gradient)*learning_rate
    b=b+(-1*b_gradient)*learning_rate
    if i%100==0:
            print('when time is:{},get_best_k:{},get_best_b:{},min_loss:{}'.format(i,k,b,min_loss))


# In[ ]:


min_loss=float('inf')
learning_rate=0.1
try_times=1000000
k=random.random()*200-100
b=random.random()*200-100
for i in range(try_times):
    y_hat=[price(x,k,b) for x in x_rm]
    current_loss=loss(y,y_hat)
    if current_loss<min_loss:
        min_loss=current_loss
    k_gradient=gradient_k(x_rm,y,y_hat)
    b_gradient=gradient_b(y,y_hat)
    k=k+(-1*k_gradient)*learning_rate
    b=b+(-1*b_gradient)*learning_rate
    if i%100==0:
            print('when time is:{},get_best_k:{},get_best_b:{},min_loss:{}'.format(i,k,b,min_loss))


# In[2]:


def linear(x,k,b):
    return k*x+b
def sigmoid(x):
    return 1/(1+np.exp(-x))
def y(x, k1,k2,b1,b2):
    output1=linear(x,k1,b1)
    output2=sigmoid(output1)
    output3=linear(output2,k2,b2)
    return output3


# In[ ]:




