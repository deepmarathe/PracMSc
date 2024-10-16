# Importing necessary 
#Design a simple machine learning model to train the training instances and test the same


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset (for this example, using the Iris dataset)
from sklearn.datasets import load_iris
data = load_iris()

# Step 2: Convert data to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Step 3: Split the dataset into training and testing sets (80% train, 20% test)
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (standardizing the data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train the KNN model
model = KNeighborsClassifier(n_neighbors=3)  # Using 3 nearest neighbors
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of KNN model: {accuracy * 100:.2f}%')
