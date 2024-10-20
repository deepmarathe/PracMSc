import random
import string

# Genetic Algorithm parameters
target_string = "HELLO"
population_size = 50  # Increased population size
mutation_rate = 0.01
generations = 200  # Increased generations for more evolution

# Fitness function: number of characters matching the target
def fitness(individual):
    return sum(1 for a, b in zip(individual, target_string) if a == b)

# Create initial population (random strings)
def create_population(size):
    return [''.join(random.choices(string.ascii_uppercase, k=len(target_string))) for _ in range(size)]

# Select parents (tournament selection)
def select_parents(population):
    tournament = random.sample(population, 5)  # Select 5 individuals instead of 3 for better diversity
    return max(tournament, key=fitness)

# Crossover (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Mutation (random character mutation)
def mutate(individual):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(string.ascii_uppercase)
    return ''.join(individual)

# Main genetic algorithm loop
population = create_population(population_size)

for generation in range(generations):
    best_individual = max(population, key=fitness)
    print(f"Generation {generation}: Best individual: {best_individual}, Fitness: {fitness(best_individual)}")
    
    if fitness(best_individual) == len(target_string):  # Stop early if the optimal solution is found
        break

    # Create new generation
    new_population = []
    for _ in range(population_size):
        parent1 = select_parents(population)
        parent2 = select_parents(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

# Best individual in the final population
best_individual = max(population, key=fitness)
print(f"Best individual: {best_individual}, Fitness: {fitness(best_individual)}")
print('Deep Marathe - 53004230016')
