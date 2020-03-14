# -*- coding: utf-8 -*-
"""
FILENAME: poly_regression.py
PROJECT: Machine Learning Algorithms
DATE CREATED: 13-Mar-20
DATE UPDATED: 13-Mar-20
VERSION: 1.0
"""
# --------------- PHASE 1: Environment Setup -------------------#
# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# intialize the data set
data = {'sno':[1,2,3,4,5,6], 'temperature':[0,20,40,60,80,100], 'pressure': [0.0002, 0.0012, 0.0060, 0.0300, 0.0900, 0.2700]}
raw_df = pd.DataFrame(data)
raw_df

X = raw_df.iloc[:, 1:2].values
y = raw_df.iloc[:, 2].values

print(X)
print(y)

# --------------- PHASE 2: Build the Model -------------------#

# fit the model
lin = LinearRegression()
lin.fit(X,y)

# Visualising the Linear Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temp') 
plt.ylabel('Pressure') 
  
plt.show() 


# build the polynomial plot
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)


# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Temp') 
plt.ylabel('Pressure') 
  
plt.show() 

# --------------- PHASE 3: Evaluate the Model -------------------#
# predict linear model
lin.predict([[110.0]])

# predict polynomial model
lin2.predict(poly.fit_transform([[110.0]]))
