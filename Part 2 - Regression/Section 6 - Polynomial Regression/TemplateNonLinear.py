#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:17:34 2018

@author: utkarshsingh
"""

""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values



#Fitting polynomial regression model to our dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_ply=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_ply,y)



#Predicting the salary using polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))

#Visualise the polynomial regression model results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(X_ply), color='blue')
plt.title('Truth or bluff polynomial regression')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()


#Visualise the polynomial regression model results with higher resolution
X_grid=np.arrange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(X_grid), color='blue')
plt.title('Truth or bluff polynomial regression')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()