# # Support Vector Regression (SVR)

""" Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" Importing the dataset"""
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
# Transform y into vertical 2D array (num rows, num cols)
y = y.reshape(len(y),1)
print(y)


""" Feature Scaling (to avoid a feature being neglected by model -> Level is too small compare to Salary)"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# standardization transform between -3 to 3
print(X)
print(y)


""" Training the SVR model on the whole dataset"""
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Radio Basis Function
# train the whole dataset
regressor.fit(X, y)


""" Predicting a new result"""
# predict feature scaled 6.5, reverse the scaled object back on the DV vector y
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


""" Visualising the SVR results"""
# scatter(position level, real salary, color)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


""" Visualising the SVR results (for higher resolution and smoother curve)"""
# By reducing the interval between each point to 0.1 rather than 1
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
# Transform X_grid into vertical 2D array
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

