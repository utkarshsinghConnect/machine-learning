#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:23:30 2018

@author: utkarshsingh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fitting linear regression model to our dataset
from sklearn.linear_model import LinearRegression
linearRegressor=LinearRegression()
linearRegressor.fit(X,y)


#Fitting polynomial regression model to our dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_ply=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_ply,y)

#Visualise the linear regression model results
plt.scatter(X,y,color='red')
plt.plot(X, linearRegressor.predict(X), color='blue')
plt.title('Truth or bluff linear regression')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()
#Visualise the polynomial regression model results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(X_ply), color='blue')
plt.title('Truth or bluff polynomial regression')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#Predicting the salary using linear regression
linearRegressor.predict(6.5)

#Predicting the salary using polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))











