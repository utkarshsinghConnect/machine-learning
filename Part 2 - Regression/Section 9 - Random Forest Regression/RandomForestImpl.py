#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 20:56:08 2018

@author: utkarshsingh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fitting our dataset to the model
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10000,random_state=0)
regressor.fit(X,y)

y_pred=regressor.predict(6.5)


#Visualising with high resolution model
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or bluff decision tree model high resolution')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()