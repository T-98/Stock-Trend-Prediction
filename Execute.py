# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 04:08:26 2018

@author: dkhar
"""
from numpy import genfromtxt
import numpy as np
import pandas as pd
""" Function will be given sliced arrays"""
data = pd.read_csv('AMZN.csv')
dataset = genfromtxt('AMZN.csv',delimiter=',')
print(dataset.shape)
