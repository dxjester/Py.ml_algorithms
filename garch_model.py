# FILENAME: garch_model.py
# PROJECT: Python ML Algorithms
# DATE CREATED: 16-Mar-20
# AUTHOR: dxjester21
# VERSION: 1.0

# import the required modules
import matplotlib.pyplot as plt
from random import gauss
from random import seed
from arch import arch_model

# set the seed
seed(1)

data = [gauss(0, i*0.01) for i in range(0,100)]

# split into train/test sets
n_tests = 10
train, test = data[:-n_tests,]

# define the model
a_model = arch_model(train, mean = 'Zero', vol = 'ARCH', p = 15)

# fit the model
a_model_fit = a_model.fit()

# forecast the test set
yhat = a_model_fit.forecast(horizon = n_tests)

# plot the variance
var = [i*0.01 for i in range(0,100)]
plt.plot(var[-n_test:])

# plot forecast variance
plt.plot(yhat.variance.values[-1, :])
plt.show()