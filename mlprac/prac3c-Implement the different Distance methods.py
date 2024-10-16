#Implement the different Distance methods (Euclidean) with Prediction, Test Score and Confusion Matrix.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2] # Select only the first two features (sepal length and sepal width)
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42)
# Initialize k-NN classifier with different distance metrics
k = 3
# List of distance metrics to test
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
# Create subplots for each distance metric
fig, axes = plt.subplots(1, len(distance_metrics), figsize=(15, 5))
for i, metric in enumerate(distance_metrics):
 knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
 # Fit the classifier to the training data
 knn_classifier.fit(X_train, y_train)
 # Make predictions on the test data
 y_pred = knn_classifier.predict(X_test)
 # Evaluate the classifier's performance
 print(f"Distance Metric: {metric}")
 print("Confusion Matrix:")
 print(confusion_matrix(y_test, y_pred))
 print("\nClassification Report:")
 print(classification_report(y_test, y_pred))
 print("\n")
 # Visualize the dataset and decision boundaries for the current metric
 ax = axes[i]
# Plot the training data points
 ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis',
label='Training Data')
 # Plot the testing data points
 ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x',
s=100, label='Testing Data')
# Plot decision boundaries using the current metric
 knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
 knn_classifier.fit(X, y)
 x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min,
y_max, 0.01))
 Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
 Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap='viridis', alpha=0.5, levels=range(4))
ax.set_title(f'K-NN ({metric.capitalize()} Metric)')
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.legend()
plt.show()