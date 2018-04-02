#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 08:48:07 2018

@author: tnvmadhav
"""

#Simple Linear regression


# Data Preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd     #import dataset

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values

#Splitting the testing and training set
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

"""#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting a simple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)