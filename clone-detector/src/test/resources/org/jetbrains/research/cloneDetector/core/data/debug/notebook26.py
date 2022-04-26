#!/usr/bin/env python
# coding: utf-8

# **This model is based on Loreau et al. 2003, PNAS.**

# In[3]:


import numpy as np
import math
from numba import jit


# In[20]:


@jit(nopython=True)
def Environmental_fluctuation(v1,t,cycle):
    EF=0.5*(np.sin(v1+2*math.pi*t/cycle)+1)
    return EF

@jit(nopython=True)
def Species_dynamics(pre,p1,p2,p3,p4,p5,Index,v1):
    v=np.delete(v1,Index)
    post=(p1*p2*p3-p4)*pre+(p5/(v1.shape[0]-1))*np.sum(v)-p5*pre
    return post

@jit(nopython=True)
def Resource_dynamics(pre,p1,p2,v1,v2):
    post=p1-p2*pre-pre*np.sum(v1*v2)
    return post

@jit(nopython=True)
def Productivity(mat1,mat2,v1,mat3):
    M,N=mat1.shape
    SCtemp=np.empty((M))
    for i in range(M):
        CPtemp=np.empty((N))
        for j in range(N):
            CPtemp[j]=mat1[i,j]*mat2[i,j]*v1[j]*mat3[i,j]
        SCtemp[i]=np.sum(CPtemp) #每个物种的生产力
    P=np.sum(SCtemp) #所有群落的总生产力
    return P


# In[ ]:


#DataType=np.float32
#######################################################
#Basic settings：
NumOfSpecies=7 #物种数量
NumOfCommunity=7 #群落数量 
T=np.int(1e5) #演化时间
DeltaT=np.int(1) #演化的时间间隔
PSC=np.empty((NumOfSpecies,NumOfCommunity)) #i物种在j群落中的数量,i为行，j为列（下同）
CSC=np.empty((NumOfSpecies,NumOfCommunity)) #i物种在j群落中消耗资源的速率
ESC=np.empty((NumOfSpecies,NumOfCommunity)) #i物种在j群落中将资源转换为生物量的效率
MSC=np.empty((NumOfSpecies,NumOfCommunity)) #i物种在j群落中的死亡速率
ASC=np.empty((NumOfSpecies,NumOfCommunity)) #i物种在j群落中的传播速率
PC=np.empty((NumOfCommunity)) #j群落中的资源量
IC=np.empty((NumOfCommunity)) #j群落中资源再生的速率
LC=np.empty((NumOfCommunity)) #j群落中资源自然流失的速率
HS=np.empty((NumOfSpecies)) #i物种的特性
EC=np.empty((NumOfCommunity)) #j群落的环境波动
XC=np.empty((NumOfCommunity)) #使EC的初值满足特定值的参数
#######################################################

#######################################################
#Special settings:
PSC=PSC+1 #i物种在j群落中的数量的初值
PC=PC+1 #j群落中的资源量的初值
Period=40000 #资源波动周期
XC=np.arcsin(2*np.arange(NumOfCommunity-1,-1,-1)/(NumOfCommunity-1)-1)
HS=np.arange(NumOfSpecies-1,-1,-1)/(NumOfSpecies-1)
HSC=np.tile(HS,(NumOfCommunity,1)).T #按群落数量将HS扩充成矩阵，牢记这是行向量扩充出的矩阵,转置后第一行为第一个物种。第二行为第二个物种，以此类推。
ESC=ESC+0.2
MSC=MSC+0.2
IC=IC+150
LC=LC+10
ASC=ASC+0.1 #物种传播速率，这里假设所有物种在所有群落中传播速率相等
#######################################################

#######################################################
#Save data
SampleInterval=100
NumOfData=T/SampleInterval+1
PData=np.empty((NumOfData)) #存储生态系统（所有群落）的总生产力
#######################################################

#######################################################
#Loop:
cb=-1
path=''
for k in range(T+1):
    EC=Environmental_fluctuation(XC,k,Period)
    CSC=1.5-np.absolute(HSC-EC) #计算出资源消耗率矩阵
    PCTemp=PC #将当前时刻的运算矩阵留存
    PSCTemp=PSC #将当前时刻的运算矩阵留存
    for j in range(NumOfCommunity):
        PC[j]=DeltaT*Resource_dynamics(PCTemp[j],IC[j],LC[j],CSC[:,j],PSCTemp[:,j])+PCTemp[j] #计算下一时刻的运算矩阵
        for i in range(NumOfSpecies):
            PSC[i,j]=DeltaT*Species_dynamics(PSCTemp[i,j],ESC[i,j],CSC[i,j],PCTemp[j],MSC[i,j],ASC[i,j],j,PSCTemp[i,:])+PSCTemp[i,j] #计算下一时刻的运算矩阵
    Pdy=Productivity(ESC,CSC,PCTemp,PSCTemp)/NumOfCommunity
    if k%SampleInterval==0: #按SampleInterval的间隔采样
        cb+=1
        filenamePC='PC_'+str(cb).zfill(5)+'.bin'
        filenamePSC='PSC_'+str(cb).zfill(5)+'.bin'
        #Data type is np.float64,矩阵维度是(NumOfSpecies,NumOfCommunity)
        PC.tofile(path+filenamePC)
        PSC.tofile(path+filenamePSC)
        PData[cb]=Pdy
filenameP='Productivity.bin'
PData.tofile(path+filenameP)

