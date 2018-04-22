#Decisiontree implementation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#Fitting the decision tree to dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Predicting the results
y_pred=regressor.predict(6.5)

#Visualising the results
#Non-continous regression models
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or bluff decision tree model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising with high resolution model
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or bluff decision tree model high resolution')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()