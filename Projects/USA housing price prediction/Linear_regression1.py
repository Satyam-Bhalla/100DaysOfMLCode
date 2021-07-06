import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = load_boston()

df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
df['target'] = dataset['target']
##Checking the dataframe
print(df.head())

##sns.pairplot(df)
##plt.show()

##Creating the training and test set
X = df.drop('target', axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

##Creating object of Linear Regression and predicting values
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

##Calculating various errors
mse = metrics.mean_squared_error(y_test, predictions)
mae = metrics.mean_absolute_error(y_test,predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test,predictions))
print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("Root Mean Square Error: ",rmse)

##Plotting the graph
plt.scatter(y_test, predictions)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()
