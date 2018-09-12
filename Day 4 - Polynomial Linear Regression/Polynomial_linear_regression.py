# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

print(dataset.head())

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

print(X,X.shape)
print(y,y.shape)

## Fitting Simple Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

## Creating polynomial features of independent variables
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

print(X_poly)

## Fitting Linear Regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

## Visualising simple linear regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Position Salary")
plt.xlabel("Position Number")
plt.ylabel("Salary")
plt.show()

## Visualising simple polynomial regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Position Salary")
plt.xlabel("Position Number")
plt.ylabel("Salary")
plt.show()

## Visualising simple polynomial regression results in more continuous way
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Position Salary")
plt.xlabel("Position Number")
plt.ylabel("Salary")
plt.show()

## predicting results for 6.5 level using linear regression
print(*lin_reg.predict(6.5))

## predicting results for 6.5 level using polynomial regression
print(*lin_reg2.predict(poly_reg.fit_transform(6.5)))