# -*- coding: utf-8 -*-
"""
FILENAME: multivariate_regression.py
PROJECT: Machine Learning Algorithms
DATE CREATED: 13-Mar-20
DATE UPDATED: 13-Mar-20
VERSION: 1.0
"""
# --------------- PHASE 1: Environment Setup -------------------#
# import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# intialize the data set
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

print(x)
print(y)

# --------------- PHASE 2: Build the Model -------------------#

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

# --------------- PHASE 3: Evaluate the Model -------------------#
# Test set value
x_new = np.arange(10).reshape((-1, 2))
print(x_new)

y_new = model.predict(x_new)
print(y_new)

