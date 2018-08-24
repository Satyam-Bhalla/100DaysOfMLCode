# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')


# Printing first five rows
print(dataset.head())

# Separating the Independent variables and Dependent variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# printing the data in X and y for checking
print(X,y)

# Checking the shape of X and y
print(X.shape,y.shape)

## Now we need to split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Creating the Linear regression object and fitting the data into model
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# Creating the predictions vector
y_pred = regressor.predict(X_test)

# Checking the results of predictions
print(y_pred,y_test)

# Plotting the linear regression line on training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Ploting the Linear regression line on test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Salary vs Experience(Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

