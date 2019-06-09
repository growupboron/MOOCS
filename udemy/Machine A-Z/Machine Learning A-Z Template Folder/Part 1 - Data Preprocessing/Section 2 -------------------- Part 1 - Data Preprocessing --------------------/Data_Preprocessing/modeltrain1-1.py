# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:55:32 2018

@author: Aman Arora
"""
"""
DATA PREPROCESSING

"""
#Preparing the Dataset for training the ML model.
#importing the libraries
#First step-numpy
import numpy as np
#contains mathematical tools, incorporates mathematics in our code.
import matplotlib.pyplot as plt
#Helps us to plot graphs etc in our code.
import pandas as pd
#Best library to import and manage datasets.


#IMPORTING DATASET
#SET UP A WORKING DIRECTORY - VERY IMPORTANT, coz it must have the data.csv file.
#We're gonna use pandas
dataset = pd.read_csv('Data.csv')
#We need to distinguish matrix of features, we have our dataset, so we create matrix of #features, so we see the independent variable columns.
X = dataset.iloc[:, :-1].values
#We take the columns of independent variables, all except the last one. : means we take #all the lines. :-1 means we take all columns except the last one.
#So we execute using highlighting this line and pressing Ctrl+Enter and type X in console to see the 
#dataset.
#Now, Create the dependent variable vector.
y = dataset.iloc[:, 3].values
#Index for the purchased column, which is dependent, i.e. y is 3, since indexes start at #0. Hence, 3 is the ending.

"""
#Sometimes, dataset is often missing. We should know how to handle it.
#Say, an info in a cell is missing. We can either delete the row, which can rather prove #costly or we #can fill in the cell with the mean of the data in the column. And this is it for almost every feature. 
#We use a library - sci-kit learn.preprocessing and we import Imputer class.
from sklearn.preprocessing import Imputer
#sklearn-amazing libraries to make ml models. preprocessing datasets can be done. Imputer allows us to #take care of the missing data.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#Press Ctrl+I to see info for class Importer on the right.
#missing_values let's us identify the missing values - see NaN.
#we use the mean strategy, the next arguement. Strategy can also be median or mode, see from the info #section (Ctrl+I when spyder highlights Import class.)
#axis: If axis=0, then impute along columns, from info.
imputer.fit(X[:, 1:3])
#*(1:3) is to include indexes 1 and 2, this is the way to write. It excludes the upper bound.
#We have taken the two columns with missing data, seen from the console when we outputted the dataset #table for X and y.
X[:, 1:3] = imputer.transform(X[:, 1:3])


#NOW, Understand Categorical variables.
#Country contains 3 categories, since it contains 3 countries, France, Spain and Germany; and Parameters #is also categorical, since it contains two categories, Yes or No.
#So, since ML models are based on equations, it might cause problems, since we only want numbers, so we #need to encode the categorical variables #to  numbers.
#We use same sklearn.preprocessing library but we import LabelEncoder class.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# labelencoder_X.fit_transform(X[:, 0]) ==> This will be changed, it will be assigned to the column.
#By typing this, we fitted labelecoder_X to the first column country of the matrix X, and this returns #the first column of country X encoded. : meanss selecting the full column, and '0' represents the first 
#column (indexing in python.)
#When we run this (Ctrl+Enter), we see that the column country has uts 3 vaules as encoded entities.
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#So we now see, X and enter in the console, we have encoded the column countries after we assigned it to #the column itself.
#THERE'S A PROBLEM
#The equations in our model may think that since Spain = 2, it's greater than France and Germany, but its #senseless. Same is the case with Germany. So we need to make sure that we  prevent the equation from #thinking one is greater than other.
#So, we use DUMMY VARIABLES.
#WHAT WE DO IN THIS IS, we take number of columns  = number of categories, i.e. 3 here, and just use 1 or #0 to represent whether the country is Germany, Spain or France.

#We use another class for this, OneHotEncoder class, from the sklearn.preprocessing library. (See above, #at the previous import).
onehotencoder = OneHotEncoder(categorical_features = [0])
#We don't really care about the " N values", the first parameter.
#We need to specify the index of the columns, i.e. the country column and its 0, so category column = [0]
X = onehotencoder.fit_transform(X).toarray()
#just X because here we specified the 0 index.
#This will OneHotEncode the cateogtical array.
#RE-EXECUTE LINE NUMBER 53, BECAUSE WE ALSO INCLUDED OneHotEncoder class, and the aboce line of code.
#Check for the Datasets using the Variable explorer and compare and see the effects after using the #OneHotEncoder.


#NOW, to take care of y, the purchased variable, the dependent one.
#This is a dependent variable, the ML model will know that its a category and there is no order between #the two.
#So, we just use the LabelEncoder.
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)
"""


#SPLITTING THE DATASET INTO TRAINING SET AND TEST SET.
#Improrting the library - cross_validation library.
from sklearn.cross_validation import train_test_split
#We create X train and X test, and then y train and  y test, the train and test parts of the dependent #and the independent variables.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#We're going to define the 4 variables at the same time.
#See first parameter in the inspect section of the train_test_split(). Putting X and y as the first #arguement in train_test_split() is like putting whole of the dataset.
#Now, the next parameter. test_size, size of the test, like, 0.5 means half of data goes to test, and #half of it goes to training set.
#Here, 10 observations, so if size = 0.2, 2 observations go into test set.
#UNDERSTANDING
#We have created two models, the train and the test. What we are doing is, we are first training the #machine to establish corelations between the dependent and the independent variables from the 8 training #sets that we did there. The machine learns from it, and then we use the test models to test on the two #sets of data whether machine can predict output correctly or not, whethre customer 0 or 1 is going to #buy the data or not.
#If the machine learns too much by heart, it will be having errors in prediction and won't be able to #understand. This will come in greater detain in regression section.


"""
Often it is automaticaly incorporated.
#FEATURE SCALING
#Very important
#Here Age and salary are not on the same scale, age is like between 20-50 and salary between 40000 to #90000. Scales are entirely different. This will cause some issues in ML model. Lot of ML models are #based on euclidian distance. sqrt((x2-x1)^2 + (y2-y1)^2). Say, age is x and salary is y. Some #euclidian distance is between observation 2 and 5, the euclidian distance will be dominated by salary, #because salary is hugely large. So in ML equations Age will like, not exist. So we bring them to same #scale.
#We use standardization.
#We import library preprocessing and import the standard_scalar class
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#We need to create another object that will scale the dependent variable vector.
#And then, when we are applying our standard scalar object to our training set, we need to fit the #object to traning set and then transform it.
X_test = sc_X.transform(X_test)
#We dont't need to fit the test set to training set since it is already fitted to training set.
#Do we need to fit and transform the dummy variables?
#It depends on our data.
#So, now when we run it, we see all the values for age and salary are in range.
#Do we need to apply feature scaling to variable y?
#No, we don't need to, coz this is a classification problem. If dependent variable take a huge set of #values, then we might need to do it.
"""