#####Code:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
# Load Dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv('C:\\Users\\deepm\\Downloads\\archive\\diabetes.csv', header=None, 
names=col_names)
pima.head()
# Split dataset into features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
Y = pima['label']
# Split dataset into training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 
##70% training 30% testing
# Create Decision Tree Classifier Object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train, Y_train)
# Predict the response for test dataset
Y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print('Accuracy:- ', metrics.accuracy_score(Y_test, Y_pred))
print('Deep Marathe')
# Visualizing Decision Trees
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,
 feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())