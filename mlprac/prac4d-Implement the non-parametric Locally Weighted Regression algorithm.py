#Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. 
#Select appropriate data set for your experiment and draw graphs.

import numpy as np
import matplotlib.pyplot as plt
# Seed for reproducibility
np.random.seed(0)
# Generate random dataset
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16))
# Locally Weighted Regression function
def locally_weighted_regression(query_point, X, y, tau=0.1):
 m = X.shape[0]
 # Calculate weights
 weights = np.exp(-((X - query_point) * 2).sum(axis=1) / (2 * tau * 2))
 W = np.diag(weights)

 # Add bias term to X
 X_bias = np.c_[np.ones((m, 1)), X]

 # Calculate theta using weighted least squares
 theta = np.linalg.inv(X_bias.T.dot(W).dot(X_bias)).dot(X_bias.T).dot(W).dot(y)

 # Predict for query_point
 x_query = np.array([1, query_point])
 prediction = x_query.dot(theta)
 return prediction
# Generate test points
X_test = np.linspace(0, 5, 100)
# Predict using locally weighted regression
predictions = [locally_weighted_regression(query_point, X, y, tau=0.1) for
query_point in X_test]
# Plot results
plt.scatter(X, y, color='black', s=30, marker='o', label='Data Points')
plt.plot(X_test, predictions, color='blue', linewidth=2, label='LWR Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
print('Deep Marathe - 53004230016')