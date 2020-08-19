#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


network=dict()
network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
network['b1']=np.array([0.1,0.2,0.3])
network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
network['b3']=np.array([0.1,0.2])
network['b2']=np.array([0.1,0.2])


# In[8]:


def sigmoid(x):
    return 1/(1+np.exp(-x))
def identity_fun(x):
    return x


# In[10]:


def forward(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=identity_fun(a3)
    return y
    
x=np.array([1,4])


# In[11]:


forward(network,x)


# In[15]:


n,d_in,h,d_out=64,1000,100,10


# In[16]:


x=np.random.randn(n,d_in)
y=np.random.randn(n,d_out)


# In[24]:


w1=np.random.randn(d_in,h)
w2=np.random.randn(h,h)
w3=np.random.randn(h,d_out)
learning_rate=1e-3
for i in range(500):
    #前向传播
    a1=np.dot(x,w1)
    z1=np.maximum(a1,0)
    a2=np.dot(z1,w2)
    z2=np.maximum(a2,0)
    y_pred=np.dot(z2,w3)
    loss=np.square(y_pred-y).sum()
    #反向传播
    gradient_ypred=2*(y_pred-y)
    gradient_w3=np.dot(z2.T,gradient_ypred)
    gradient_z2=np.dot(gradient_ypred,w3.T)
    gradient_a2=gradient_z2.copy()
    gradient_a2[a2<0]=0
    gradient_z1=np.dot(gradient_a2,w2.T)
    gradient_w2=np.dot(z1.T,gradient_a2)
    gradient_a1=gradient_z1.copy()
    gradient_a1[a1<0]=0
    gradient_w1=np.dot(x.T,gradient_a1)
    #参数更新
    w3-=learning_rate*gradient_w3
    w2-=learning_rate*gradient_w2
    w1-=learning_rate*gradient_w1
    print("epoch:{},loss={},w1={},w2={},w1={}".format(i,loss,w1,w2,w3))


# In[ ]:




