# FILENAME: bank_dataset.py
# PROJECT: Support Vector Machine Application
# DATE CREATED: 19-OCT-19
# DATE UPDATED: 19-OCT-19
# VERSION: 1.0

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)

# --------------------------- PHASE 1: DATA IMPORT ---------------------------#
bank_raw = pd.read_csv("bill_authentication.csv")

bank_raw.columns # print column names

bank_raw.shape # get dimensions of the dataframe

bank_raw.head(20) # display the top 20 records

bank_raw['Class'].unique() # only two Response values [0,1]

bank_data = bank_raw.copy() # create a copy of the dataframe

# display initial findings
bank_data.shape
bank_data.describe()



# --------------------------- PHASE 2: DATA PLOT -----------------------------#
bank_pairplot = sns.pairplot(bank_data, hue="Class")



# --------------------------- PHASE 3: SVM SETUP -----------------------------#

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

