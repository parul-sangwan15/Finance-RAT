__author__ = 'Arcady Dushatsky'

from csv import DictReader
import collections
from sklearn import cross_validation
from Plots import plot_auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np

#logistic regression implementation
def logistic():

    #Read data
    data = pd.read_csv('train.csv')
    data.fillna(0, inplace=True)

    #Multiply positive objects 19 times
    positive_mult = 19
    data_1 = data[data.SeriousDlqin2yrs.isin([1])]
    pieces = list([])
    pieces.append(data)
    for i in range(positive_mult):
        pieces.append(data_1)
    data1 = pd.concat(pieces)

    #Multiply data 3 times
    epoches = 3
    pieces = list([])
    for i in range(epoches):
        pieces.append(data1)
    data = pd.concat(pieces)
    data = data.applymap(float)
    print(data.shape)

    #Process data
    column_names = list(data)
    column_names.remove('SeriousDlqin2yrs')
    column_names.remove(column_names[0])
    X = data[column_names]
    Y = data['SeriousDlqin2yrs']

    #Create training validation datasets, size of validation is 0.1
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)

    #Create Logistic Regression with Regularization
    inverse_regularizetion_value = 0.03
    log = LogisticRegression(C = inverse_regularizetion_value)

    #Fit Logistic regression
    log.fit(X_train, y_train)

    #Make predictions on validation set
    pred_ = log.predict_proba(X_val)
    pred = list([])
    for p in pred_:
        pred.append(p[1])
    pred = np.array(pred)

    #Calculate Loss, AUC and plot ROC-curve
    print(log_loss(np.array(y_val), np.array(pred)))
    plot_auc(np.array(y_val), np.array(pred))

    #Print regression coefficients
    print(log.coef_)

    #Read and process test dataset
    test_data = pd.read_csv('test.csv')
    test_data.fillna(0, inplace=True)
    column_names = list(test_data)
    column_names.remove('SeriousDlqin2yrs')
    column_names.remove(column_names[0])
    test_data = test_data[column_names]

    #Make predictions on test dataset
    pred_ = log.predict_proba(test_data)
    pred = list([])
    for p in pred_:
        pred.append(p[1])
    pred = np.array(pred)

   

logistic()
