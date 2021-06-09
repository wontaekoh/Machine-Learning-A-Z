# # Polynomial Regression

""" Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" Importing the dataset"""
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


""" Training the Simple Linear Regression model on the whole dataset"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


""" Training the Polynomial Regression model on the whole dataset"""
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


""" Visualising the Linear Regression results"""
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


""" Visualising the Polynomial Regression results"""
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


""" Visualising the Polynomial Regression results (for higher resolution and smoother curve)"""
# By reducing the interval between each point to 0.1 rather than 1
X_grid = np.arange(min(X), max(X), 0.1)
# Transform X_grid into vertical 2D array
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


""" Predicting a new result with Linear Regression"""
# [[]]: two dimension array
# first[](outer): row, second[](inner): column
# ex, [[5, 4], [3, 2]] ->
# [5, 4]
# [3, 2]
lin_reg.predict([[6.5]]) # first row, first column


""" Predicting a new result with Polynomial Regression"""
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))



