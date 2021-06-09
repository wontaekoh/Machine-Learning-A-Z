# # Data Preprocessing Tools

""" Importing the libraries """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" Importing the dataset """
dataset = pd.read_csv('Data.csv')
# features/independent variables (all the values except the last attribute)
X = dataset.iloc[:, :-1].values # index location[rows, columns]
# dependent variables/class(result)
y = dataset.iloc[:, -1].values
print(X)
print(y)


""" Taking care of missing data """
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])  # 2nd to 3rd columns
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)


""" Encoding categorical data """
# # Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# transforming first columne of the table, others remains intact
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


""" Splitting the dataset into the Training set and Test set """
from sklearn.model_selection import train_test_split
# test set = 20% of dataset, 80% for train set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)


""" Feature Scaling (to put all the features in the same scale)"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Apply standardisation on numerical values
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)