# FILENAME: svm_main.py
# PROJECT: Support Vector Machine Application
# DATE CREATED: 19-OCT-19
# DATE UPDATED: 19-OCT-19
# VERSION: 1.0

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------- PHASE 1: DATA IMPORT ---------------------------#


# 1.a: Import the bank data set ----------------------------------------------#
bank_raw = pd.read_csv("bill_authentication.csv")

# get dimensions of the dataframe
bank_raw.shape

# display the top 20 records
bank_raw.head(20)

# create a copy of the dataframe
bank_data = bank_raw.copy()



# 1.b: Import the iris data set ----------------------------------------------#
# import IRIS data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# create column names and consolidate in a list
col_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

iris_data = pd.read_csv(url, name=col_names) # Read dataset to pandas dataframe




# --------------------------- PHASE 2: SVM SETUP -----------------------------#

# separate the dataframe between the four (4) x predictor variables from the one (1) x respons variable
x_predict = bank_data.drop('Class', axis = 1)
y_response = bank_data['Class']

# split the data set to train, test, validate sets
x_train, x_test, y_train, y_test = train_test_split(x_predict, y_response, test_size = 0.20)

# print out shapes of the x & y train and test sets
x_train.shape
x_test.shape
y_train.shape
y_test.shape

# apply the linear SVM kernel as the data classifier
svm_classifier = SVC(kernel = 'linear')
svm_classifier.fit(x_train, y_train)

# create the predictor variables
y_pred = svm_classifier.predict(x_test)
y_pred

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
