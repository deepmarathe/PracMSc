import pandas as pd

# Load the penguins dataset from a CSV file
df = pd.read_csv('C:\\Users\\deepm\\Deep\\msprac\\AAIJounral\\penguins (1).csv')

# Preview the data
print("Data Preview:")
print(df.head())

# Create a pivot table for joint probability
# Pivot table will be for Species (rows) and Island (columns), and we'll compute frequencies
pivot_table = pd.crosstab(df['species'], df['island'], normalize=True)

print("\nJoint Probability (Pivot Table):")
print(pivot_table)

# Example: Conditional Probability of Species given Island
# We can normalize along columns to get conditional probabilities
conditional_probability = pivot_table.div(pivot_table.sum(axis=0), axis=1)

print("\nConditional Probability of Species given Island:")
print(conditional_probability)

# To calculate Joint Probability, we already have it in the pivot table,
# normalize=True gives joint probabilities
print("\nJoint Probability is represented in the pivot table (Species vs Island):")
print(pivot_table)

# Example: Calculating P(Species = Adelie | Island = Biscoe)
p_adelie_given_biscoe = conditional_probability.loc['Adelie', 'Biscoe']
print(f"\nP(Adelie | Biscoe) = {p_adelie_given_biscoe:.4f}")

# Author information
print('Deep Marathe - 53004230016')
