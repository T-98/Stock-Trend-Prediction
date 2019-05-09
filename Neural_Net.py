# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:50:11 2017

@author: Divyansh Khare
"""
import numpy as np
from numpy import genfromtxt
import pandas as pd

#sigmoid function
def sig_func(x):
    return 1/(1+np.exp(-x))
#sigmoid derivative
def sig_deriv(x):
    return sig_func(x)*(1 - sig_func(x))

#Creating Dataset 
seeds_data = pd.read_csv("seeds_binary.csv")
dataset = genfromtxt('seeds_binary.csv',delimiter=',')
feat = dataset[:,0:6]
trget = dataset[:,7,None]
input_neurons = feat.shape[1] 
hidden_layerN = 3 
output_layerN = 1
X = feat
#print X
y = trget
#---------------DUMMY DATASET-------------------------
"""feat = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])
input_neurons = feat.shape[1] 
hidden_layerN = 3 
output_layerN = 1
X = feat"""
#----------------------------------------


#Initialising weight matrices
b12 = np.random.randn(1,hidden_layerN)
b23 = np.random.randn(1,output_layerN)
W12= np.random.randn(input_neurons,hidden_layerN)
W23= np.random.randn(hidden_layerN,output_layerN)
neta = 0.1
for i in range(6000):
    #Forward Pass
    h_in = np.dot(X,W12) + b12
    h_out = sig_func(h_in)
    #h_out = np.insert(h_out,0,np.ones((1,h_out.shape[0])),1)
    o_in = np.dot(h_out,W23) +b23
    o_out = sig_func(o_in)
    
    
    #Backward Pass
    #for W23
    """Why a negative sign was requires ??"""
    Err = -(y - o_out)
    dell23 = Err*sig_deriv(o_out)
    delta23 = np.dot(h_out.T,dell23)
    #for W12
    dell12 = sig_deriv(h_out)
    theta = np.dot(dell23,W23.T)*dell23
    delta12 = np.dot(X.T,theta)
    #updating the weights
    b12 = b12 - dell12
    b23 = b23 - dell23
    W12 = W12 - neta*delta12
    W23 = W23 - neta*delta23
for i in range(o_out.shape[0]):
    print o_out[i] , "--",y[i]
