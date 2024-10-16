#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
#Load California Hosuing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.DataFrame(housing.target, columns=['MEDV'])
#Visualise dataset correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of California Hosing Features")
plt.show()
#Split the dataset into training and testingf sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42)
#Implement Least Squares Regression (Linear Regression)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
#Make predictions
y_train_pred = reg_model.predict(X_train)
y_test_pred = reg_model.predict(X_test)
#Generate relevant metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
#Print the results
print(f"Training Mean Squared Error: {train_mse}")
print(f"Test Mean Squared Error: {test_mse}")
print(f"Training R^2 Score: {train_r2}")
print(f"Test R^2 Score: {test_r2}")
# Visualise regression coefficients
coefficients = pd.DataFrame(reg_model.coef_.T, X.columns,
columns=['Coefficient'])
print(coefficients)
#Plot predicted vs actual values for test set
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, c='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Test Set)")
plt.show()
print("Deep Marathe - 53004230016")