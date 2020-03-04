# FILENAME: svm.py
# PROJECT: Machine Learning Project
# DATE UPDATED: 4-MAR-20
# VERSION: 1.0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

# define scatter points
x_data = [1, 5, 1.5, 8, 1, 9]
y_data = [2, 8, 1.8, 8, 0.6, 11]

# create array to store data points
X = np.array([[1,2],
	     [5,8],
	     [1.5,1.8],
	     [8,8],
       	     [1,0.6],
	     [9,11]]) 

# target (response) values
y = [0,1,0,1,0,1]

# build the scatterplot
plt.scatter(x_data,y_data)
plt.show()

# create the svm classifier
classifier = svm.SVC(kernel = 'linear', C=1.0)

# fit the linear model
classifier.fit(X,y)

# predict outcomes for test values
test1 = np.array([[0.58, 0.76]])
print(classifier.predict(test1))

# predict outcomes for test values
test2 = np.array([[10.58, 10.76]])
print(classifier.predict(test2))

w = classifier.coef_[0]
print(w)

# display graph
a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - classifier.intercept_[0]/w[1]

plt.figure(figsize = (8,6))
h0 = plt.plot(xx, yy, 'k-', label = "non weighted div")

plt.scatter(x_data, y_data)
plt.legend()
plt.show()