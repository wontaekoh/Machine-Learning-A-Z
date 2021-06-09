# # Multiple Linear Regression

""" Importing the libraries """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" Importing the dataset """
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)


""" Encoding categorical data """
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# index = 3 as categorical data (state) is at 4th column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


""" Splitting the dataset into the Training set and Test set """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


""" Training the Multiple Linear Regression model on the Training set """
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


""" Predicting the Test set results """
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2) # display 2 decimals 
# concat two vectors and display vertically 
# reshape(num of rows, num of column)
# to compare predicted and real profits
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


"""Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')"""
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

"""Therefore, our model predicts that the profit of a Californian startup which spent 160000 in R&D, 130000 in Administration and 300000 in Marketing is $ 181566,92.

Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array. Simply put:
1,0,0,160000,130000,300000→scalars 
[1,0,0,160000,130000,300000]→1D array 
[[1,0,0,160000,130000,300000]]→2D array """


"""Getting the final linear regression equation with the values of the coefficients"""
print(regressor.coef_)
print(regressor.intercept_)

"""Therefore, the equation of our multiple linear regression model is:

Profit = 86.6 × Dummy State 1 − 873 × Dummy State 2 + 786 × Dummy State 3 + 0.773 × R&D Spend + 0.0329 × Administration + 0.0366 × Marketing Spend + 42467.53

Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values."""



