# -*- coding: utf-8 -*-
"""
FILENAME: nn_back_propagation.py
PROJECT: Machine Learning Algorithms
DATE CREATED: 13-Mar-20
DATE UPDATED: 13-Mar-20
VERSION: 1.0
"""
# import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# intialize the data set
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
print(x)

y = np.array([5,20,14,32,22,38])
print(y)

plt.scatter(x, y, color= 'purple')
plt.show()

# build the linear regression model
linear_mod = LinearRegression()

# fit the model
model = LinearRegression().fit(x,y)

# retrieve the R-squared value
r_square = model.score(x,y )
print('Coefficient of determination: ', r_square)

# retrieve intercept
print('Intercept: ', model.intercept_)

# retrieve slope
print('Slope: ', model.coef_)

# predict
y_pred = model.predict(x)
print('Predicted Response: ', y_pred)

# Test set value
x_test = np.arange(15).reshape((-1,1))
print(x_test)

y_test = model.predict(x_test)
print(y_test)


# print the linear refression line
plt.scatter(x, y, color= 'purple')
plt.plot(x, y_pred, color ='red', linewidth=2)
plt.show()
