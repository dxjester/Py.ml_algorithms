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

bank_raw.columns # print column names

bank_raw.shape # get dimensions of the dataframe

bank_raw.head(20) # display the top 20 records

bank_data = bank_raw.copy() # create a copy of the dataframe

# display initial findings
bank_data.shape
bank_data.describe()

# 1.b: Import the iris data set ----------------------------------------------#
# import IRIS data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# create column names and consolidate in a list
col_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# read in the csv file to a pandas dataframe
iris_raw = pd.read_csv(url, names=col_names) 

# display the top 20 records
iris_raw.head(20)

# create a copy of the iris dataframe
iris_data = iris_raw.copy()

# display initial findings
iris_data.shape
iris_data.describe()


# --------------------------- PHASE 2: SVM SETUP -----------------------------#

# 2.a: bank data SVM set up --------------------------------------------------#
# separate the dataframe between the four (4) x predictor variables from the one (1) x respons variable
x_predict_bank = bank_data.drop('Class', axis = 1)
y_response_bank = bank_data['Class']

# split the data set to train, test, validate sets
x_train_bank, x_test_bank, y_train_bank, y_test_bank = train_test_split(x_predict_bank, y_response_bank, test_size = 0.20)

# print out shapes of the x & y train and test sets
x_train_bank.shape
x_test_bank.shape
y_train_bank.shape
y_test_bank.shape

# apply the linear SVM kernel as the data classifier
svm_classifier = SVC(kernel = 'linear')
svm_classifier.fit(x_train_bank, y_train_bank)

# create the predictor variables
y_pred_bank = svm_classifier.predict(x_test_bank)
y_pred_bank

print(confusion_matrix(y_test_bank,y_pred_bank))
print(classification_report(y_test_bank,y_pred_bank))

# 2.b: iris data SVM set up --------------------------------------------------#

# separate the dataframe between the four (4) x predictor variables from the one (1) x respons variable
x_predict_iris = iris_data.drop('Class', axis = 1)
y_response_iris = iris_data['Class']

# split the data set to train, test, validate sets
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(x_predict_iris, y_response_iris, test_size = 0.20)

# print out shapes of the x & y train and test sets
x_train_iris.shape
x_test_iris.shape
y_train_iris.shape
y_test_iris.shape

# apply the linear SVM kernel as the data classifier
svm_classifier = SVC(kernel = 'linear')
svm_classifier.fit(x_train_iris, y_train_iris)

# create the predictor variables
y_pred_iris = svm_classifier.predict(x_test_iris)
y_pred_iris

print(confusion_matrix(y_test_iris,y_pred_iris))
print(classification_report(y_test_iris,y_pred_iris))