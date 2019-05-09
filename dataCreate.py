# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:54:26 2017

@author: dkhar
"""
from numpy import genfromtxt
import numpy as np
import pandas as pd
from sklearn import svm 
import csv
import os
"""extracts file path"""
rootNSE = '.'
fileNSE = os.listdir(rootNSE)
# dataroot = os.path.join(rootNSE,fileNSE[2])
dataroot="GOOG.csv"
""" Function will be given sliced arrays"""
data = pd.read_csv(dataroot)
dataset = genfromtxt(dataroot,delimiter=',')
dataset = dataset[2:,1:]

"""Moving Average"""
def CalcMA(x):    
    return np.average(x)

"""Weighted 10 day Moving average calculator"""
def CalcWavg(x):
    w = np.array([10,9,8,7,6,5,4,3,2,1])
    w = w.reshape((10,1))
    x = w*x
    return np.average(x)
    

"""Calculates momentum"""
def CalcMom(x):
    return (x[0,0] - x[9,0])
    
"""Stochastic K%"""
def CalcK(x):
    x = np.reshape(x,(x.shape[0],1))
    m=np.amin(x)
    M=np.amax(x)
    K = (x[0,0] - m)/float((M-m))
    return K*100
    
"""Stochastic D%"""
def CalcD(K,end=1300):
    D = np.zeros((0,0))
    for i in range(K.shape[0]):
        if i<end:
            k = K[i:i+10,0]
            D = np.append(D,np.average(k))
    return D

"""William R%"""
def CalcR(x,period=10):
    m = np.amin(x)
    M = np.amax(x)
    return ((M-x[0,0])/(M-m))*(-100)

"""Commodity Channel Index"""
#calculates typical price

def Truncate(x,beg= 1301):
    x = np.delete(x,np.s_[beg:],0)
    x = np.reshape(x,(beg,1))
    return x
    
def CalcDt(x,y):
    Dt = abs(x-y)
    Dt = np.reshape(Dt,(Dt.shape[0],1))
    return Dt

def CalcCCI(x,period=10):
    TP = np.zeros((0,0))
    Tavg =np.zeros((0,0))
    avg =np.zeros((0,0))
    Tsma = np.zeros((0,0))
    prices = x[1:,2:5]
    """Calculating TP"""
    for i in range(prices.shape[0]):
        Tavg =(prices[i,0]+prices[i,1]+prices[i,2])/3
        TP = np.append(TP,Tavg)
    TP = np.reshape(TP,(TP.shape[0],1))
    """Calculating TPSma"""
    for i in range(TP.shape[0]):
            sma = TP[i:i+period,0]
            avg = np.average(sma)
            Tsma = np.append(Tsma,avg)
    Tsma = np.reshape(Tsma,(Tsma.shape[0],1))
    """Calculating CCI"""
    Dt = 0.015*CalcDt(TP,Tsma)
    Truncate(TP)
    Truncate(Tsma)
    Truncate(Dt)
    diff = TP - Tsma
    return diff/Dt

"""Accumulation/Distribution Oscillator (Input:High,Low,Current)"""
def CalcAD(x):
    prices = x[1:,2:5]
    ADO = np.zeros((0,0))
    for i in range(prices.shape[0]):
        if i<1301:
            #if (prices[i,0] - prices[i,1]) != 0:   
            AD = (prices[i,0] - prices[i+1,2])/(prices[i,0]-prices[i,1]) 
            #else:
            #    AD = (prices[i,0] - prices[i+1,2])/(0.00000001)
            ADO = np.append(ADO,AD)
    return ADO
"""Relative Strength index RSI"""
def CalcAvgs(x):
    change = np.zeros((x.shape[0],1))
    gain = np.zeros((0,0))
    loss = np.zeros((0,0))
    for i in range(x.shape[0]):
        if i <x.shape[0]-1:        
            change[i,0] = x[i,0] - x[i+1,0];
        if change[i,0] >= 0:
            gain =np.append(gain,change[i,0])
            gain = gain.reshape((gain.shape[0],1))
        else:
            loss =np.append(loss,change[i,0])
            loss = loss.reshape((loss.shape[0],1))
    return gain,loss
def CalcRSI(x,period=14):
    RSI = np.zeros((0,0))
    for i in range(x.shape[0]):
        if i <1844:            
            param = x[i:i+period,0,None]
            g,l = CalcAvgs(param)
            avgG = np.average(g)
            avgL = np.average(l)
            RS = avgG/avgL
            mid = 100 - (100/RS)
            RSI = np.append(RSI,mid)
            RSI = np.reshape(RSI,(RSI.shape[0],1))
    return RSI
"""Creating Dataset for the predictive model"""

ClosingP = dataset[1:,4,None]
prices = dataset[1:,2:5]
MA = np.zeros((0,0))
WMA = np.zeros((0,0))
K = np.zeros((0,0))
R = np.zeros((0,0))
for i in range(ClosingP.shape[0]):  
    Input = ClosingP[i:i+10,0] 
    MA = np.append(MA,CalcMA(Input))
    WMA = np.append(WMA,CalcWavg(Input))
    K = np.append(K,CalcK(Input))
    R = np.append(R,Input)
    #TP = CalcTP(Input)
    #TPsma = CalcTPsma(Input)
    #Dt = CalcDt(Input)
Rsi = CalcRSI(ClosingP)
"""Reshaping the returned arrays"""
Input = np.reshape(Input,(Input.shape[0],1))
MA = np.reshape(MA,(MA.shape[0],1))
WMA = np.reshape(WMA,(WMA.shape[0],1))
K = np.reshape(K,(K.shape[0],1))
R = np.reshape(R,(R.shape[0],1))
"""-----------------------------------------------------"""
CCI = CalcCCI(dataset)
ADO = CalcAD(dataset)
D =CalcD(K,end=1855)
"""-----------Truncating every parameter to same size---------------"""
MAf = Truncate(MA)
WMAf =Truncate(WMA)
Kf =Truncate(K)
Rf = Truncate(R)
ADOf =Truncate(ADO)
Df = Truncate(D)
RSIf = Truncate(Rsi)

Y = np.zeros((0,0))
for i in range(ClosingP.shape[0]-1):
    if ClosingP[i,0] >= ClosingP[i+1,0]:
        Y = np.append(Y,1)
    else:
        Y = np.append(Y,0)
Y = np.reshape(Y,(Y.shape[0],1))
#print MAf.shape,WMAf.shape,Kf.shape,Rf.shape,ADO.shape,Df.shape,RSIf.shape
features = np.concatenate([MAf,WMAf,Kf,Rf,ADOf,Df,RSIf],1)
Y = Truncate(Y,beg=1300)
features = np.delete(features,(0),axis=0)
print (features.shape)
print (Y.shape)
FinalData = np.concatenate([features,Y],1)

np.savetxt('finalData.csv',FinalData,delimiter=',')

X_train = features
Y_train = Y
X_test = features[0,:]
Y_test = Y[0,:]

