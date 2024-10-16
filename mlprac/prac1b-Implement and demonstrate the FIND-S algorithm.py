import pandas as pd

# Step 1: Load training data from CSV file
# Example data file content:
# Outlook,Temperature,Humidity,Wind,PlayTennis
# Sunny,Hot,High,Weak,No
# Sunny,Hot,High,Strong,No
# Overcast,Hot,High,Weak,Yes
# ...

# Replace 'your_data.csv' with your actual CSV file
data = pd.read_csv("C:\\Users\\deepm\\Deep\\msprac\\mlprac\\your_data.csv")

# Step 2: Initialize the most specific hypothesis
def find_s_algorithm(data):
    # Extract positive examples (where the target is 'Yes')
    positive_examples = data[data['PlayTennis'] == 'Yes'].drop('PlayTennis', axis=1).values

    # Initialize hypothesis with the first positive example
    hypothesis = positive_examples[0]

    # Step 3: Generalize hypothesis based on other positive examples
    for example in positive_examples[1:]:
        for i in range(len(hypothesis)):
            if hypothesis[i] != example[i]:
                hypothesis[i] = '?'  # Generalize the hypothesis

    return hypothesis

# Step 4: Find the most specific hypothesis
hypothesis = find_s_algorithm(data)
print(f'The most specific hypothesis is: {hypothesis}')
