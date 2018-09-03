# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# Importing dataset
dataset = pd.read_csv('50_Startups.csv')

# Printing first five rows
print(dataset.head())

# Separating the Independent variables and Dependent variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


print("X contains the following values: \n",X)
print("y contains the following values: \n",y)

# Checking the shape of X and y
print(X.shape,y.shape)

# Now we will convert the categories into numbers from 1st and last column of the Dataset
labelEncoder_x = LabelEncoder()
# labelEncoder_x is a object of LabelEncoder class which converts text to numbers
X[:, 3] = labelEncoder_x.fit_transform(X[:,3])
# Now we will use one hot encoder to expand the columns corresponding to the number of categories
oneHotEncoder = OneHotEncoder(categorical_features= [3])
X = oneHotEncoder.fit_transform(X).toarray()
print(X)


## Now we will avoid the Dummy Variable Trap
X = X[:,1:]

print(X)

## Now we need to split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)
print(y_pred)


## ****************Backward Elimination Technique*******************
## Building an optimal model using backward elimination i.e we will remove all those columns which are not giving us any accurate
## Predictions
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
print(X)

## Now we will remove all the independent variables whose value is greater than the significance value
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()