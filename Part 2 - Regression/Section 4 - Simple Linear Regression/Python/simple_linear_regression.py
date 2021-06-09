# # Simple Linear Regression

""" Importing the libraries """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" Importing the dataset """
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


""" Splitting the dataset into the Training set and Test set """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


""" Training the Simple Linear Regression model on the Training set """
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


""" Predicting the Test set results """
y_pred = regressor.predict(X_test)


""" Visualising the Training set results """
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


""" Visualising the Test set results """
plt.scatter(X_test, y_test, color = 'red')
# same regression line as training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


""" Making a single prediction (for example the salary of an employee with 12 years of experience) """
print(regressor.predict([[12]]))

"""
Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138531.

Important note: Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:

12→scalar 
[12]→1D array 
[[12]]→2D array
"""


""" Getting the final linear regression equation with the values of the coefficients """
print(regressor.coef_)
print(regressor.intercept_)
""" Therefore, the equation of our simple linear regression model is:
  DV   = constant + (coefficient x IV)  
Salary = 26780.10 + (9312.58 × YearsExperience)

Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values. """
