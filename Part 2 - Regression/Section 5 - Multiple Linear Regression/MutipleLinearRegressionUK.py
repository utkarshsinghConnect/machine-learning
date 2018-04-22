#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:06:08 2018

@author: utkarshsingh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X=LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X=X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set result
y_pred=regressor.predict(X_test)

#Building the optimal model using Backward elimination
# We are using this library because this library does not include the x0 element and that is goig to help us learn tmachine learning

import statsmodels.formula.api as sm
X= np.append(arr=np.ones((50,1)).astype(int),values=X, axis=1)

#store only those matrix which has high impact as per rules
# All the independent variable
#X_opt=X[:,[0,1,2,3,4,5]]
#
#regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#regressor_OLS.summary()
#
#X_opt=X[:,[0,1,3,4,5]]
#
#regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#regressor_OLS.summary()
#
#
#
#X_opt=X[:,[0,3,4,5]]
#
#regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#regressor_OLS.summary()
#
#
#X_opt=X[:,[0,3,5]]
#
#regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#regressor_OLS.summary()
#
#
#X_opt=X[:,[0,3]]
#
#regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#regressor_OLS.summary()

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL=0.05
X_opt=X[:,[0,1,2,3,4,5]]
X_modeled=backwardElimination(X_opt,SL)





