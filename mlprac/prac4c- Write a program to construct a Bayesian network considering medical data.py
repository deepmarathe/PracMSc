#Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the 
#diagnosis of heart patients using standard Heart Disease Data set.


import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator,MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
data = pd.DataFrame (data={'Age': [30, 40, 50, 60, 70],
 'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
 'ChestPain': ['Typical', 'Atypical', 'Typical', 'Atypical',
'Typical'],
 'HeartDisease': ['Yes', 'No', 'Yes', 'No', 'Yes']})
model = BayesianNetwork([('Age', 'HeartDisease'),
 ('Gender', 'HeartDisease'),
 ('ChestPain', 'HeartDisease')])
model.fit(data, estimator=MaximumLikelihoodEstimator)
pos = nx.circular_layout(model)
nx.draw(model, pos, with_labels=True, node_size=5000,
node_color="skyblue", font_size=12, font_color="black")
plt.title("Bayesian Network Structure")
plt.show()
for cpd in model.get_cpds():
 print("CPD of", cpd.variable)
 print(cpd)
inference = VariableElimination(model)
query = inference.query(variables=['HeartDisease'], evidence={'Age':50,
'Gender': 'Male', 'ChestPain': 'Typical'})
print(query)
print("Deep Marathe -53004230016")