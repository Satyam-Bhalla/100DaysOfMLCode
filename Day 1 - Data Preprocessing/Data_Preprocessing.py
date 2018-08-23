# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing Dataset
dataset = pd.read_csv('Data.csv')
print(dataset.head())

# First three columns are independent variables
# storing values of independent variables in X
X = dataset.iloc[:,:-1].values

# Last column is Dependent variable so we store this in y variable
y = dataset.iloc[:,3].values

print("Independent values: ",X)
print("Dependent values: ",y)

## Taking care of Missing Data From Dataset
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print("Now the missing data is removed: ",X)

# Now we will convert the categories into numbers from 1st and last column of the Dataset
labelEncoder_x = LabelEncoder()
# labelEncoder_x is a object of LabelEncoder class which converts text to numbers
X[:, 0] = labelEncoder_x.fit_transform(X[:,0])

print("Now the data in the first column is: ",X[:,0])


# But the machine learning models will think that Germany has a higher precedence
# than France and spain because the value of germany is 2
# So to remove this issue we will create three columns for each column
# the value will be one corresponding to that country in the dataset
oneHotEncoder = OneHotEncoder(categorical_features= [0])
X = oneHotEncoder.fit_transform(X).toarray()

print("Now X has 5 columns: ",X)

# Now we will convert the categories into numbers from 1st and last column of the Dataset
labelEncoder_y = LabelEncoder()
# labelEncoder_x is a object of LabelEncoder class which converts text to numbers
y = labelEncoder_y.fit_transform(y)

print("Now the dependent data is transformed into numbers: ",y)

## Now we need to split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Feature Scaling

#  We do feature scaling because most of the machine learning algorithms works
#  upon the Euclidean distance p=sqrt((x2-x1)^2 - (y2-y1)^2)
#  In this case we can see salary is a dominating feature which will effect our results
#  So we will do feature scaling to improve this thing
#  It can be done in two ways: -> Standardisation and Normalisation.
#  ### Standardisation is x = (x-mean(x))/standard deviation(x)
#  and 
#  ### Normalisation is x = (x-min(x))/max(x)-min(x)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("Now we will check the columns of X_train and X_test",X_train,X_test)
