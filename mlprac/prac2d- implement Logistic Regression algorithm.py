#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.datasets import load_breast_cancer
#Step 1: Load the inbulit Breast Cancer dataset
cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
#Feature
y = pd.DataFrame(cancer_data.target, columns=['target']) # Target
#Step 2: Explore the dataset
print("Dataset Head:")
print(X.head()) # Preview the first few rows of the feature set
print("\nTarget Distribution:")
print(y['target'].value_counts()) # Distribution of the target variable (0 =malignant 1= benign)
#Step 3: Split the databaset into traning and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
#Step 4 : Implement Logistic Regression
logreg = LogisticRegression(max_iter=10000, random_state=42) #Incresed max_itr to ensure convergence
logreg.fit(X_train, y_train.values.ravel()) # y_train must be passed as a flat aaray
#Step 5 ; Make predictions on the test set
y_pred = logreg.predict(X_test)
#Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
# Step 7: Visualize the Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matric')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
#Step 8:Make a prediction on a new input sample
#Example:Let's create a new sample input (using the mean of feature for simplicity)
#You can replace these values with actual feature values you'd like to predict for
new_input = np.array([X.mean().values])
# Ensure the new input has the correct shape (1, n_features)
print(f"\nNew Input for Prediction:\n{new_input}")
#Make a prediction on the new input
new_prediction = logreg.predict(new_input)
#Get the predictied class (0= malignant, 1=bening)
predicted_class = 'bening' if new_prediction == 1 else 'malignant'
print(f"\nPredited class for the new input: {predicted_class}")
#Step 9 : Visualise the Confusion Matrix for the test set
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix- Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
print('Deep Marathe - 53004230016')