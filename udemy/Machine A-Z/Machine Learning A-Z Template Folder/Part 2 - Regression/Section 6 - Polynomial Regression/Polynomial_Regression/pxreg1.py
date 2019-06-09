# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:16:40 2018

@author: Aman Arora
"""

#===================================POLYNOMIAL REGRESSION========================================

#We take the data preprocessing template.
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Understanding the dataset : We're a HR team for a company, and we're about to make an offer to a potential new employee. There's his data, which the HR wants to check, his different salaries in the different jobs he did before coming to this company. Now, using Excel, we observe, there is a kinda non linear relationship, kinda a polynomial relationship between the two parameters, salary and his Level of job. 
#This new employee had been a region manager for 2 years now, and usually, it takes on average 4 years from being a region manager to a partner. So this employee was halfway between level 6 and 7, so we can say 6.5, since he told that his salary was previously 1,60,000. So now this HR tells to team that he can predict using regression models, whether he is bluffing about his salary or not.


#Now, looking at the dataset, we see that Position has alredy been encoded by the column Level, so we don't need to bring in the concept of categorical variables. So we don't include the column Position in the martrix of features.
"""
X = dataset.iloc[:, 1].values

Now see here, if we just write iloc[:, 1], its fine, but the only problem is that the matrix of features, X is taken in as a vector, and not a matrix. So, if we want it to be taken as a matric, we just specify the column as 1:2, done below, since python does not include the upper bound.

"""
X = dataset.iloc[:, 1:2].values #So, now X is a matrix with 10 rows and 1 column.
y = dataset.iloc[:, 2].values


#NOW : Below is the section for splitting the dataset into training set and test set, but, in this dataset, it is not required, since we have a very less number of observation, that we'd like to divide to the 2 sets and the model will be highly inaccurate. Also this might affect the negotiation of HR team with the candidate employee. So, all of the observations will be used to train the model.


# Splitting the dataset into the Training set and Test set
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
"""


# Feature Scaling : No need.
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) 
"""


#FITTING LINEAR REGRESSION TO THE DATASET.
#We're creating both models for comparison.
from sklearn.linear_model import LinearRegression
#We're going to make 2 regressors, 1 for linear LinReg for linear, and LinReg2 for polynomial regression.
lin_reg = LinearRegression() #No need for any arguement.
lin_reg.fit(X, y)
#We didn't do any splitting, so just X and y are the two arguements of the fit() method.


#POLYNOMIAL REGRESSION MODEL.
#We import a new class PolynomialFeatures from sklearn.preprocessing library.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)  #Creating object of the PolynomialFeatures class.
#Transformer tool that'll transform the matrix of features X to a new matrix, X_poly, containing x1, x1^2 and even higher powers if we want.
#Degree is the first arguement, specifies the degree.
X_poly = poly_reg.fit_transform(X)
#Since we are transforming the matrix of features X to X_poly, we use the fit_transform() method.
#If we visit variable explorer, we compare X and X_poly. The X_poly has 3 columns, x0, column of 1's, constant in the multiple linear regression equation, column of x, the original, and then the column of squared numbers. Now we have to include this fit. We create a new LinearRegression object, lin_reg2.
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
#We fitted y, dependent variable, to X_poly in this step. X_poly contains two independent variables.


#VISUALIZING THE LINEAR REGRESSION RESULTS.
plt.scatter(X, y, color = 'red') #Real observation points.
#Real levels and real salaries.
plt.plot(X, lin_reg.predict(X), color = 'blue')
#We input X in the parenthesis, we are predicting salaries of the 10 position levels, which are contained in the X matrix, so we input X.
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
#Clearly we see that the linear graph is not good, if we speak of predictions.
#If we bring up the cursor to level, x = 6.5, we see that the predicted salary is around US$330,000, If this employee hadn't said anything about his salary, he'd have got US$330,000 as his salary and the employee would have been happy. 

#So, a better model is clearly required.


#VISUALIZING THE POLYNOMIAL REGRESSION.
plt.scatter(X, y, color = 'red')

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

"""
#Since there isno polynomial term.
plt.plot(X, lin_reg2.predict(X_poly), color = 'blue')
"""
#We DON'T have to use X_poly in the second arguement, because X_poly is already defined as transformation of some matrix of feature, X. If we have a new matrix of features X here, since X_poly is already defined, if we want to add new observations we just use poly_reg.fit_transform(). It generalizes the model for any X, any matrix of features.
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Truth or Bluff (POLYNOMIAL Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#US$194,421 is the salary now that we get from the polynomial regression model, at level of 6.5 approximately.
#To make this even better, we add a degree to our polunomial regression model.
#We just do a change in line 67, we change degree = 2, to degree = 3. We can keep on increasing it, to increase the accurace. I have the degree = 5, which gives the results, almost equal to US$161000, the best prediction, at 6.5 level.
#This model will be used here, since it is hoghly accurate.

#BUT STILL, since there are some straight lines in between points, we need to make the graph more continous. SDo, for that, what we do is we use another function of the numpy library, the arrange(), with the variable X_grid, that will contain all the levels  + incremented steps between the levels with a resolution 0.1 .


#Predicting a new result with linear regression 
lin_reg.predict(6.5) #To predict the previous salary of employee
#We used this method to predict value of salary for just 1 value of X, ie 6.5 unlike previously where we entered the full matrix of features, X inside the predict method.

#Predicting result according to Polynomial regression model
lin_reg2.predict(poly_reg.fit_transform(6.5))
#Actually, the verdict is true, and almost accurate. The comany will recruit this honest employee.





























































