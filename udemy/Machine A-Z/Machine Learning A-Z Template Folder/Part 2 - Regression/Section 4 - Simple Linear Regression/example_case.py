# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 20:23:53 2018

@author: Aman Arora
"""

#New example for trial.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('example_case.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(y)
y = imputer.transform(y)
"""
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2, random_state = 0)
"""
#Again feature scaling is not required.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)

regressor.predict(10)
"""
#Drawing the scatter (Graph)

plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Example_test case training')
plt.xlabel('Values of X')
plt.ylabel('Values of y')
plt.show()

plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'orange')
plt.title('Example test case testing')
plt.xlabel('Values of x')
plt.ylabel('Values of y')
plt.show()

"""
































