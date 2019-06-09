# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 21:00:06 2018

@author: Aman Arora
"""

#MULTIPLE LINEAR REGRESSION.
#Several independent variables.

#Prepare the dataset.
#We use the previous template for the job of preparing the dataset.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values #The profit columns is fourth one based on counting is python.

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #TO CHANGE THE TEXT INTO NUMBERS. 3rd columns.
onehotencoder = OneHotEncoder(categorical_features = [3]) #Index of column is 3.
X = onehotencoder.fit_transform(X).toarray()
# We don't need to encode the Dependent Variable.

#Avoiding the dummy variable trap.
X = X[:, 1:]
#I just removed the first column from X. By doing that I am taking all lines of X, but then by putting #'1:'. I want to take all columns of X starting from 1 to end. I dont't take the first column, to avoid #the dummy variable trap. Still python library automatically takes care of the dummy variable trap.

# Splitting the dataset into the Training set and Test set
#10 observations in test set and 40 in training set make a good split => 20% or 0.2 of the dataset is #test_size.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling, not necessary, the library will do for us.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#The thing to understand here is that we are going to build a model to see if there's some linear #dependencies between all the independent variables and the dependent variable profit. The model should predict the profit based on the marketing spend, r n d, etc. variable.

#Our matrix of features is, since we know going to conatin the independent variables, is going to contain columns R&D, Administration, marketing and state spends, which is the matrix of features, or of independent variables, X. y here is going to be the last column, profit.

#In the matrix of features we will have to encode the column state, since it has categorical variables, various states. So we will be using the OneHotEncoder. It's done above, before splitting dataset into #training and test set.


#FITTING THE MULTIPLE LINEAR REGRESSION TO THR TRAINING SET.
from sklearn.linear_model import LinearRegression
#Coz we are still making lineaar regression, but with various variables.
regressor = LinearRegression()
#Fitting this object to the training set.
regressor.fit(X_train, y_train) #I fit the multiple linear regressor to the training set.
#We are now going to test the performance of the multiple linear regression model on the test set.


#Final round.
#If we see our dataset, we have 4 independent variables, and 1 dependent variable. If we wanted to add a #visual step to plot a graph, we'd need 5 dimensions. So, we proceed to predictions of test results.
#Creating the vector of predictions.
y_pred = regressor.predict(X_test)
#Now see and compare the y_test and y_pred test sets.

#What if among these various independent variables, we want to select those which have high, or which have #low impact on the dependent variable, called statistically significant or insignificant variables. The #goal now is to find a team of independent variables which are highly staistically significant.


#BACKWARD ELIMINATION.
#We will need a library : stats models formula API library, sm (stats models)
import statsmodels.formula.api as sm
#We just need to add a column of 1's in the matrix of independent variables, X. Why? Becuase see, multiple linear regression equation, in it, there's a constant b0, which is as such not associated with any independent variable, but we can clearly associate it with x0 = 1. (b0.x0 = b0).
#The library we've included does not include this b0 constant. So, we'll somehow need to add it, because #the matrix of features X only includes the independent variables. There is no x0 = 1 anywhere in the #matrix of features and its most lbraries, its definitely included. But this is not with statsmodel #library. So we've to add the column of 1's, otherwise, the library will think the equation is #b1x1+b2x2+____bnxn.
#Doing so using append function from numpy library.
"""
X = np.append(X, values = np.ones((50, 1)).astype(int), axis = 1)
"""
#If we inspect the array function, first parameter is arr, or our matrix of features, X. The next parameter is values, that in this case we want 1. So, we need to add an array, as written, column of 1's. #The array is a matrix of 50 lines and 1 column, with only 1 value inside. It's very easy with a function of numpy called ones() which creates an array of only 1's value inside. We need to specify the number of lines and column. The first arguement of the ones() is shape, which lets us set-up the array. We input (50, 1) to create array of 50 rows and 1 column, as the first argument.
#To prevent the datatype or 'dtype' error, we just convert the matrix of features to integer, using astype() function. 
#The third argument of the append function is axis, if we want to add a column, axis = 1, and if we want a row, axis = 0. 
#Now, this will add the column on the end. But, what we want is to add the column in the beginning, to maintain systematicity. We just need to invert the procedure, that is, add the matrix of features X to the single column we have made! I comment out the above line and rewrite it below.
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)


#STARTING WITH BACKWARD ELIMINATION.
#We've prepared the backward elimination algorithm.

#First thing to do is to create a new matrix of features, which is going to be optimal matrix of features, #X_opt.
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#This is going to be in the end, containing only the independent variables that are statistically significant. As we know, BE consists of including all independent variables, then we go on excluding independent variables. We're going to write specifically all indexes of columns in X. Coz we are going to remove various indexes. The indexes are included as [0, 1, 2,3 , 4, 5].
#Now we will select the significance level. We choose it as 5%.
#Step2 : Fit the full model with all possible predictors, that we just did above. But we haven't fitted it yet. Doing that now.
#We use a new library, statsmodel library, and we create a new regressor which is the object of new class, which is called, OLS, ordinary least squares.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#endog is the dependent variable, y.
#exog is the array with no. of observations and no. of regressions, X_opt or matrix of features, and its not included by default. Intercept needs to be added by the user, or me.

#Step3
#We've to look for the predictor which has the highest P value, that is the independent variable with highest P value. Read notebook. We use a function of statsmodel library summary() which returns a great table containing the matrix that helps make our model more robust like r^2, etc. and we'll get all the P values, we'll compare it to the significance level to decide whether to remove it from our model or not.
regressor_OLS.summary()
#We are interested in the P value, which is the probability, and when we run this line, we get a great table which shows all the important parameters needed to checkout for backward elimination. We'll see about R-squared and Adj. R-squared later, to make our matrix more robust. P value is the probability, lower the P value, more significant the independent variable is wrt dependent variable. x0 = 1, x1 and x2 the two dummy variables for state. Then x3 is for R&D, and x4, for min spend, and x5 for marketing spend. 
#Now, we clearly see, One with highest P value is x2, with 99%. So, we are way above significance level of 5%, so we remove it according to step 4. 
#If we see for matrix X from variable explorer, we see that the variable x2 or the one with index 2 is one of the dummy variables. So we will remove it.
X_opt = X[:, [0, 1, 3, 4, 5]] #Removed the variable with highest probability.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Fitted the model w/o the variable x2.
regressor_OLS.summary()

#Now, we see, variable x1 has the highest P value, which is greater than 5%, which is 94%. So we'll remove it.
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Again.

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Again.

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Here, I am left with just R&D spend and the constant. Its a very small P value, and so, R&D spend is a powerful predictor for profit, and has a high effect on the dependent variable on the profit. Whether we needed to remove marketing spend or not, since it was very near to the significance level, will be seen in the next to come algorithms for ML.
#====================THIS IS THE BACKWARD ELIMINATION ALGORITHM=========================


    




    















































































