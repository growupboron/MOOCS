# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:56:15 2018

@author: Aman Arora
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#X is the matrix of features, taking the dataset and removing the lsat column of the dataset, ie, the salary #column.
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #Creating the Independent variable vector.
y = dataset.iloc[:, 1].values #Creating the dependent variable vector.
#The last or the dependent column is the 2nd one, so index is 1.
#y here is a vector, and X is a matrix. ===VVV IMP===

#Splitting the dataset into the Training set and Test set
#20 in train set and 10 in test is good.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#Random state is so that the teacher and I have same result.

# Feature Scaling
#Not required as of now.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#Most of the libraries that we'll use in simpole linear regression, the libraries are going to take care #of the feature scaling.


#FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET.
#First is to import the sklearn library "linear model" and import class LinearRegression, then we will #create the object of this class, regressor, and so call this as it is, we will fit it to training set, #for that we use method "fit()", fits the regressor object that we create to the training set.
from sklearn.linear_model import LinearRegression
#Creating object.
regressor = LinearRegression()
#() because like a function that will return an object of itself.
#The parameters that are shown are not that needed so we don't write anything in the parentheseis.
regressor.fit(X_train, y_train)
#So, here the machine is the simple linear regression model and the learning is the fact that this slr #machine learns on the training set here.
#Most simple machine learning model.
#The regressor learned the relations between the dependent and independent variable and it will now #predict the results now.


#PREDUCTING THE TEST RESULTS.
#We will create a vector of predictive values, that will contain the predictions of the test set salary #and we will put all these predicted saaries into a vector y_pred, the vector of predictions.
y_pred = regressor.predict(X_test)
#As the parameter, we input the matrix on which we perform the testing, the X_test, see Ctrl+I for more #information on the module predict().
#And now we compare the prediction made by the model and the actuaal values, ie comparison between y_test #and y_pred.


#Visualizing the TRAINING set. Seeing how the linear dependency is.
#We use matplotlib.pyplot library which is already imported.
#To plot employees no. of years (x) vs salary (y).
#We use plt.scatter which ,akes the scatter plot.
plt.scatter(X_train, y_train, color = 'yellow')
#PARAMETERS : X coordinate, X_train, the real no. of years; then y_train, the real salaries;  
#Now we plot the regression line, or the predictions.
plt.plot(X_train, regressor.predict(X_train), color = 'orange')
#X coord will be X_train for x coord of the line, and y coord of the regression line will be the #predictions of the X_train, its NOT y_pred, coz it predicts based on the X_test, we are comparing real #salaries.
#Adding the title or label now.
plt.title('Salary vs Experience(Training set)') #title of the plot
plt.xlabel('Years of experience of the employees') #label for x axis.
plt.ylabel('Salaries of the employees') #label for y axis.
plt.show() #show() to fininsh and make the graph ready.
#In the plot, for eg, for years of experience = 4, actual salary is approx 55000, if we join the yellow #point to the regression line we see the predicted salary for employee having experience = 4 years is #approx. 60000.
#Now we plot the test set observation points.

#Visualizing the TEST set results.
plt.scatter(X_test, y_test, color = 'red') #Plots our observation points.
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#Our regressor is trained on the train set, so we need not change it, if we do so, we will obtain some #new regression equation, and it will be somthing different.
plt.title('Salary vs Experience(Test set)') #title of the plot
plt.xlabel('Years of experience of the employees') #label for x axis.
plt.ylabel('Salaries of the employees') #label for y axis.
plt.show() #show() to finish and make the graph ready.
#So we train the machine, which is the simple linear regression model and learning is that we trained #this model on the training set, that it learned some corelations of the training set to be able to make #some future predictions.



























































