import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
#Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
#Split the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
#Define a simple rule-based classifier function
def rule_based_classifier(x):
    if x[2] < 2.0:
        rule = "If feature 2 < 2.0, assign to Classd 0"
    return 0 # Class 0 
    elif x[3] > 1.5:
    rule = "If feature 2 >= 2.0 and feature 3 > 1.5, assign to Class 2"
    return 2 # Class 2
    else:
    rule = "If feature 2 >= 2.0 and feature 3 <=1.5, assign to Class 1"
    return 1 # Class 1
    print("Rule:", rule)
# Apply the rule-based classifier to make predictions on the test set
y_pred = [rule_based_classifier(x) for x in X_test]
# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred,
target_names=iris.target_names)
# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
print("Gayatri Kulkarni - 53004230002")
