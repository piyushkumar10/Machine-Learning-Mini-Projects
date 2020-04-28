# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Geography", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Backward Elimination

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
xopti=np.array(X[:,[0,1,2,3,4,5]],dtype=float)
optimised=sm.OLS(endog=y,exog=xopti).fit()
print(optimised.summary())

#Remove the Index with highest P Value
xopti=np.array(X[:,[0,1,3,4,5]],dtype=float)
optimised=sm.OLS(endog=y,exog=xopti).fit()
print(optimised.summary())
#Remove the Index with highest P Value

xopti=np.array(X[:,[0,3,5]],dtype=float)
optimised=sm.OLS(endog=y,exog=xopti).fit()
print(optimised.summary())
#Remove the Index with highest P Value

xopti=np.array(X[:,[0,3,5]],dtype=float)
optimised=sm.OLS(endog=y,exog=xopti).fit()
print(optimised.summary())
#Remove the Index with highest P Value

xopti=np.array(X[:,[0,3]],dtype=float)
optimised=sm.OLS(endog=y,exog=xopti).fit()
print(optimised.summary())