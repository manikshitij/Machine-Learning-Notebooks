# Machine Learning Algorithm #1 : Linear Regression
# Author : Anuradha C

# Program to compute Umbrella Sales prediction based on Rainfall Data
# Using Linear Regression Least Sqares ML Algorithm

import pandas as pd
import numpy as np

from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from scipy import stats

# Read training and test data
df_train = pd.read_csv('C://Users/anu/workspace/Python/ML_BD/train.csv')
df_test = pd.read_csv('C://Users/anu/workspace/Python/ML_BD/test.csv')

#Extract the Rainfall and Sales data lists from training data set 
x_train = df_train['Rainfall']
y_train = df_train['Sales']

#Extract the Rainfall and Sales data lists from test data set
x_test = df_test['Rainfall']
y_test = df_test['Sales']

# Convert Dataframes to Numpy ND Arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

# Apply LinearRegression Least Sqares Function
clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)

# y_pred holds the array of predicted Sales values
y_pred = clf.predict(x_test)

# r2 Score is a measure of accuracy of prediction.
# r2 compares predicted y and test y. If result is close to 1, then prediction is accurate
print('The prediction accuracy = ', r2_score(y_test,y_pred))

# Round the Ubmrella Sales figure into int
y_pred_rounded = np.around(y_pred,0)

# Reconvert the array values into a DataFrame, to write into CSV file
df_predict = pd.DataFrame({'Rainfall' : x_test.tolist(), 'Predicted Sales': list(y_pred_rounded.tolist())}, columns=['Rainfall', 'Predicted Sales'])

df_predict.to_csv('C://Users/anu/workspace/Python/ML_BD/predict.csv', encoding='utf-8', index=False)

# Calculate slope 'm' and intercept 'c' of the regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(x_train.flatten(),y_train.flatten())
print ("The slope and intercept of the regression line are :" , slope,intercept)

# Apply prediction model to any user given input of Rainfall data
x_user = float(input('Enter the Rainfall value : '))
m = np.asscalar(slope)
c = np.asscalar(intercept)

y_result = m*x_user + c
print ("Predicted Umbrella Sales = ", round(y_result))

# Plot outputs as a scatter, and the regression line passing through
plt.title('Test Sales Data Vs Predicted Sales ')
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()