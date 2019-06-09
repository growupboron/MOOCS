#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 19:16:54 2018

@author: aman
"""


#===========================SVR - SUPPORT VECTOR REGRESSION===============================
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling

#We know, most libraries have feature scaling included in their algorithms, but, here we used SVR class, which is quite uncommon and does not seem to have that algorithm included. So we need to apply feature scaling by ourselves, otherwise, we get a straight line for regression, which predicts inaccurate result.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.array(y).reshape(-1, 1)

#I needed this extra step, because of this error : 
"""
ValueError: Expected 2D array, got 1D array instead:
array=[-0.72004253 -0.70243757 -0.66722767 -0.59680786 -0.49117815 -0.35033854
 -0.17428902  0.17781001  0.88200808  2.64250325].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
"""
#So, I used reshape method for array method of numpy library, as directed by the python kernel itself to avoid the error.

y = sc_y.fit_transform(y)
#RECAP, We created two objects of StandardScaler class, sc_X and sc_y, which scales our matrix of features X and dependent vector y. We apply the fit_transform() method, each of these two objects need to be fitted to a certain matrix, one to X and one to y, so we can't create a single object.



# Fitting the SVR Model to the dataset

#Support vector machine - this is what SVR is, for regression, so we import SVR class from sklearn.svm library.
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#Here, if we see object inspection, the most important parameter is the "kernel", default value is rbf, Specifies the kernel type, linear, polynomial or gaussian svr, to be used in the algorithm. We want rbf kernel, because we know that our problem is non linear here, so we shouldn't use linear kerner which'd make a linear ml model. poly and rbf can both work, though. rbf is the gaussian kernel. That's all we need to do here as of now.
regressor.fit(X, y)
#This creates an SVR regressor.



# Create your regressor here

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#We applied feature scaling to our data here. So, we need to retransform it. Since, 6.5 is in some way not transformed, we apply the sc_X object, which is used to scale the features, we apply this to 6.5 so that it can be suited to the regressor.

#HENCE WE GET A GREAT PREDICTION, US$170,370 FOR LEVEL 6.5 .

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#There is an exception for US$1mn salary. The reason for that is that the svr model has some penalty parameters selectedby default, since CEO observatio point is quite far from other observation points, and so the model fits it near to the other points, making just the last opoint inaccurate.




































