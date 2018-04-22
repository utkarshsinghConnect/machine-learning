#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 01:39:06 2018

@author: utkarshsingh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the datasets
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,3]

#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

