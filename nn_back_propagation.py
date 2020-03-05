# -*- coding: utf-8 -*-
"""
FILENAME: nn_back_propagation.py
PROJECT: Machine Learning Algorithms
DATE CREATED: 4-Mar-20
DATE UPDATED: 4-Mar-20
VERSION: 1.0
"""

import numpy as np

# declare Sigmoid function
def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# initialize the dataset
X = np.array([
                [0,0,1],
                [0,1,1],                
                [1,0,1],
                [1,1,1]
        ])

# output dataset
y = np.array([[0,0,1,1]]).T

# define seed
np.random.seed(1)

# initialized random weight values, resulting in mean = 0
synapse0 = 2 * np.random.random((3,1)) - 1

for iter in range(10000):
    
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, synapse0))
    
    # calculate error
    l1_error = y - l1
    
    # multiply how muchg we mssed by the slope of the Sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)
    
    # updated weights
    synapse0 += np.dot(l0.T, l1_delta)
    
print("Output after training: ", l1)
    
    