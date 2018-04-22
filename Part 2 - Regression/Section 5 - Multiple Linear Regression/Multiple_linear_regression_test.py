#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:41:09 2018

@author: utkarshsingh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""import the dataset
iloc function actually splits the dataset and give us the input and output variable 
which can be used for machine learning methods
here first variable in iloc represents number of rows and second variable
represents number of columns
"""
dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

"""
Encode the categorical variable
our dataset column 3 contains the state names, now machine algorithms needs
numerical value for everything hence here we need to encode the state column 
such that it can be represented in mathematical models, also at the same time
numbers representing the state should be used for only representational purpose
and cannot be treated as 2 greater than 1 fashion.
LabelEncoder class just convert the unique labels into numeric values starting from
0 and OneHotEncoder class is used for converting those values into matrix of
the same unique numbers into 0 and 1 so that they work as switch and not actual
number, this helps in mathematical operations which is being done in machine
learning models
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()

"""Avoiding the dummy variable trap"""
X=X[:,1:]

"""Splitting the dataset into training and test set
please note that cross_validation module is not deprecated for train_test_split
and model_selection module should be used instead of this
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

"""Fitting multiple linear regression to the training set
LinearRegression class can be used for training the machine here
we pass the training set data to our regressor object and once training is done
then we call regressor predict method with test set data and visualise the
differences between the training and test set data
"""
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

"""
building the model using backward elimination
statsmodel does not identify or take the b0 constant of multiple linear regression model
that means y= b0+ b1*x1 + b2*x2 etc here we need to add the b0 element to our model
"""
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
"""Here we are just representing the same X with : representing all rows and 
0 to 5 representing the number of columns
why are we doing this?-> because we are building model using backward elimination hence we
must know which column needs to be removed when it goes beyond the SL limit of 0.05
initialised with original matrix of independent variable
"""
X_opt=X[:, [0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()
"""
When you call the summary method it prints the P value and after looking
at the highest P value which is greater than 0.05 we remove and refit
the model
"""
X_opt=X[:, [0,1,3,4,5]]
regressor_OLS=sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()


X_opt=X[:, [0,3,4,5]]
regressor_OLS=sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

















