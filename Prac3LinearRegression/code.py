import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Load Dataset
diabetes = datasets.load_diabetes()
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y = True)
# Description of the dataset
print(diabetes['DESCR'])
print(diabetes.feature_names)
# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]
# Split the data intro training and testing datasets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# Split the targets into training and testing datasets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing sets
diabetes_y_pred = regr.predict(diabetes_X_test)
# The Coefficients
print('Coefficients : \n', regr.coef_)
# The mean squared error
print("Mean Squared Error: %.2f" % mean_squared_error(diabetes_y_test,diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test,diabetes_y_pred))
# plot the outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='blue')
plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.title('Linear Regression')
plt.show()

