import pandas as pd
col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
# Load Dataset
pima = pd.read_csv('diabetes.csv', header=None, names=col_names)
pima.head()
#########Output:
# Split dataset in features and target variable
feature_cols = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
Y = pima.label # Target variable
# Split X and Y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=16)
# import the class
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=16)
# fit the model with data
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test,Y_pred)
cnf_matrix
#####################Output:
# Visualizing confusion matrix using HeatMap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create HeatMap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
####################\Output:
from sklearn.metrics import classification_report
target_names = ['Without Diabetes', 'With Diabetes']
print('Classification Report:-')
print(classification_report(Y_test,Y_pred,target_names=target_names))
###################################Output:
# ROC Curve
Y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred_proba)
auc = metrics.roc_auc_score(Y_test, Y_pred_proba)
plt.plot(fpr, tpr, label = 'data 1, auc = '+str(auc))
plt.legend(loc=4)
plt.show()
print("Deep Marathe")
