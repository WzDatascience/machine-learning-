#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# In[2]:


sos_token=0
eos_token=1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


class Vocabulary(object):
    def __init__(self):
        self.word2idx={}
        self.idx2word={0:"<sos>",1:"<eos>",-1:"<unk>"}
        self.idx=2
    def add_word(self,word):
        if not word in self.word2idx:
            self.word2idx[word]=self.idx
            self.idx2word[self.idx]=word
            self.idx+=1
            
    def add_sentence(self,sentence):
        for word in sentence.split():
            self.add_word(word)
            
    def __call__(self,word):
        if not word in self.word2idx:
            return -1
        return self.word2idx[word]
    def __len__(self):
        return self.idx

class EncoderRnn(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderRnn,self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
    
    def forward(self,inputs,hidden):
        embedded=self.embedding(inputs).view(1,1,self.hidden_size)
        #将embedding 向量作为循环网络输输入，得到一个输出和隐藏层
        output,hidden=self.gru(embedded,hidden)
        return hidden
    def sample(self,seq_list):
        word_inds=torch.LongTensor(seq_list).to(device)
        h=self.initHidden()
        for word_tensor in word_inds:
            h=self(word_tensor,h)
        return h
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)
    
class decoder(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(decoder,self).__init__()
        self.hidden_size=hidden_size
        self.maxlen=10
        self.embedding=nn.Embedding(output_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)
        
    def forward(self,seq_input,hidden):
        output=self.embedding(seq_input).view(1,1,-1)
        output=F.relu(output)
        output,hidden=self.gru(output,hidden)
        output=self.softmax(self.out(output[0]))
        return output,hidden
    
    def sample(self,pre_hidden):
        inputs=torch.LongTensor([sos_token],device=device)
        hidden=pre_hidden
        res=[sos_token]
        for i in range(self.maxlen):
            output,hidden=self(inputs,hidden)
            topv,topi=output.topk(1)
            if topi.item()==eos_token:
                res.append(eos_token)
                break
            else:
                res.append(topi.item())
            inputs=topi.squeeze().detach()
        return res


# In[4]:


#将CMN文件读取进去
with open('C:/Users/zhouwei/Desktop/L9/cmn.txt','r',encoding='utf-8')as f:
    file=f.readlines()
temp=[]
for line in file:
    temp.append(line[:-1].split("\t"))
data=[]
for n in temp:
    data.append(n[:-1])
data


# In[5]:


def sentence2tensor(lang, sentence):
    indexes = [lang(word) for word in sentence.split()]
    indexes.append(eos_token)
    return torch.tensor(indexes, dtype=torch.long,device=device).view(-1, 1)
def pair2tensor(pair):
    input_tensor = sentence2tensor(lan1, pair[0])
    target_tensor = sentence2tensor(lan2, pair[1])
    return (input_tensor, target_tensor)
lan1 = Vocabulary()
lan2 = Vocabulary()
data = [['Hi .', '嗨 。'],
        ['Hi .', '你 好 。'],
        ['Run .', '跑'],
        ['Wait !', '等等 ！'],
        ['Hello !', '你好 。'],
        ['I try .', '让 我 来 。'],
        ['I won !', '我 赢 了 。'],
        ['I am OK .', '我 沒事 。']]


for i,j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)
print(len(lan1))
print(len(lan2))

# 定义Encoder和Decoder以及训练的一些参数
import random
learning_rate = 0.001
hidden_size = 256

# 将Encoder, Decoder放到GPU
encoder = EncoderRnn(int(len(lan1)),hidden_size).to(device)
decoder = decoder(hidden_size,int(len(lan2))).to(device)
# 网络参数 = Encoder参数 + Decoder参数
params = list(encoder.parameters()) + list(decoder.parameters())
# 定义优化器
optimizer = optim.Adam(params, lr=learning_rate)
loss = 0
# NLLLoss = Negative Log Likelihood Loss
criterion = nn.NLLLoss()
# 一共训练多次轮
turns = 200
print_every = 20
print_loss_total = 0
# 将数据random choice，然后转换成 Tensor
training_pairs = [pair2tensor(random.choice(data)) for pair in range(turns)]

# 训练过程
for turn in range(turns):
    optimizer.zero_grad()
    loss = 0
    x, y = training_pairs[turn]
    input_length = x.size(0)
    target_length = y.size(0)
    # 初始化Encoder中的h0
    h = encoder.initHidden()
    # 对input进行Encoder
    for i in range(input_length):
        h = encoder(x[i],h)
    # Decoder的一个input <sos>
    decoder_input = torch.LongTensor([sos_token]).to(device)
    
    for i in range(target_length):
        decoder_output, h = decoder(decoder_input, h)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, y[i])
        if decoder_input.item() == eos_token:break
                
    print_loss_total += loss.item()/target_length
    if (turn+1) % print_every == 0 :
        print("loss:{loss:,.4f}".format(loss=print_loss_total/print_every))
        print_loss_total = 0
        
    loss.backward()
    optimizer.step()

# 测试函数
def translate(s):
    t = [lan1(i) for i in s.split()]
    t.append(eos_token)
    f = encoder.sample(t)   # 编码
    s = decoder.sample(f)   # 解码
    r = [lan2.idx2word[i] for i in s]    # 根据id得到单词
    return ' '.join(r) # 生成句子
print(translate('我们 一起 打 游戏 。'))


# In[ ]:




