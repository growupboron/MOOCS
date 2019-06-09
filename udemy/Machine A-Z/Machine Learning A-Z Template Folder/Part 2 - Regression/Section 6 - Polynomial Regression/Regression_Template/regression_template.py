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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#We removed the linear_model, since it proves to be inaccurate.

#So, we intend to fit the polynomial regression to the dataset, and hence we name it simply as Regression, ince only this polynmial regression is being used.

#We'll be removing various sections from the code itself to make the template ready.


#Fitting the regression model to the dataset.

#CREATE YOUR REGRESSOR HERE.

#Predicting a new result with polynomial regression.
y_pred = regressor.predict(6.5)
#We had this y_pred variable, which is the predicted salary of the 6.5 level, and not the vector ofpredictions (A single value).


#Visualizing the Polynomial regression results.
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
#Here above, we'ss use our regressor object, and we use predict() method to predict the salaries of all the position levels contained in oiur matrix of features, X.
plt.title('Regression')
plt.xlabel()
plt.ylabel()
plt.show()

#For higher resolution levels and smoother curve.
#Visualizing the Polynomial regression results.
X_grid = np.arange(min(X), max(X), 0.1)
#min(X), level 1, and max(X), level 10.
#This gives us a vector, but we want a matrix since X is most often a matrix.
X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#Here above, we'ss use our regressor object, and we use predict() method to predict the salaries of all the position levels contained in oiur matrix of features, X.
plt.title('Regression')
plt.xlabel()
plt.ylabel()
plt.show()