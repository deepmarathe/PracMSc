#For a given set of training data examples stored in a 
#.CSV file, implement and demonstrate the CandidateElimination algorithm to output a description of the set of all 
#hypotheses consistent with the training examples



import pandas as pd

# Load training data from CSV file
data = pd.read_csv('C:\\Users\\deepm\\Deep\\msprac\\mlprac\\prac1d.csv')

# Initialize the specific boundary (S) to the most specific hypothesis
def initialize_s(data):
    return ['ϕ'] * (len(data.columns) - 1)  # The most specific hypothesis

# Initialize the general boundary (G) to the most general hypothesis
def initialize_g(data):
    return [['?'] * (len(data.columns) - 1)]  # The most general hypothesis

# Check if a hypothesis is consistent with a given example
def is_consistent(hypothesis, example):
    for h, e in zip(hypothesis, example):
        if h != '?' and h != e:
            return False
    return True

# Candidate Elimination Algorithm
def candidate_elimination_algorithm(data):
    # Step 1: Initialize S and G
    S = initialize_s(data)
    G = initialize_g(data)

    # Step 2: Iterate through each training example
    for index, row in data.iterrows():
        example = row[:-1]  # Features
        label = row[-1]     # Class label (Yes/No)

        # Step 3: Update S and G based on the example
        if label == 'Yes':  # Positive example
            # Remove inconsistent hypotheses from G
            G = [g for g in G if is_consistent(g, example)]

            # Generalize S if needed
            for i in range(len(S)):
                if S[i] == 'ϕ':  # Update S to the first positive example
                    S = list(example)
                elif S[i] != example[i]:
                    S[i] = '?'  # Generalize S to handle positive example

        elif label == 'No':  # Negative example
            # Remove inconsistent hypotheses from S
            if is_consistent(S, example):
                S = initialize_s(data)

            # Specialize G if needed
            for g in G:
                for i in range(len(g)):
                    if g[i] == '?':
                        g[i] = example[i]

    return S, G

# Step 4: Run the Candidate Elimination Algorithm
S, G = candidate_elimination_algorithm(data)

# Output the result
print("Final Specific Boundary (S):", S)
print("Final General Boundary (G):", G)
