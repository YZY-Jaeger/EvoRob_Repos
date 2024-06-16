import numpy as np
from functions import fitness

# Initialize the population with random values within given bounds
def initialize_population(size):
    return np.random.uniform(-32.768, 32.768, (size, 3))

def tournament_selection(population, fitnesses, num_parents, tournament_size=3):
    selected = []
    for _ in range(num_parents):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        best_participant = participants[np.argmax(fitnesses[participants])]
        selected.append(population[best_participant])
    return np.array(selected)

# Perform simulated binary crossover (SBX) between two parents
def sbx_crossover(parent1, parent2, eta=1.0, prob_crossover=0.9):
    children1, children2 = parent1.copy(), parent2.copy()
    if np.random.rand() < prob_crossover:
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                beta = np.random.uniform(-0.1, 0.1)
                children1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                children2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                children1[i] = np.clip(children1[i], -32.768, 32.768)
                children2[i] = np.clip(children2[i], -32.768, 32.768)
    return children1, children2

# Mutate an individual by altering its genes slightly
def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.1):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, sigma)
            individual[i] = np.clip(individual[i], -32.768, 32.768)
    return individual

# Replace the old population with the new one while maintaining some of the best
def elitist_replacement(old_population, offspring, old_fitnesses, offspring_fitnesses, elite_size=5):
    combined_population = np.vstack((old_population, offspring))
    combined_fitnesses = np.concatenate((old_fitnesses, offspring_fitnesses))
    indices = np.argsort(-combined_fitnesses)
    new_population = combined_population[indices[:len(old_population)]]
    return new_population

def evolutionary_algorithm(pop_size, num_generations=100, mutation_rate=0.1, prob_crossover=0.9, 
                           tournament_size=3, elite_size=5):
    population = initialize_population(pop_size)
    fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in population])

    best_fitnesses = []
    avg_fitnesses = []

    for generation in range(num_generations):
        fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in population])
        parents = tournament_selection(population, fitnesses, pop_size, tournament_size)
        offspring = []
        num_parents = len(parents) - 1
        for i in range(0, num_parents, 2):
            child1, child2 = sbx_crossover(parents[i], parents[i + 1], prob_crossover=prob_crossover)
            offspring.append(child1)
            offspring.append(child2)
        offspring = np.array(offspring)
        mutated_offspring = np.array([gaussian_mutation(ind, mutation_rate=mutation_rate) for ind in offspring])
        offspring_fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in mutated_offspring])

        population = elitist_replacement(population, mutated_offspring, fitnesses, offspring_fitnesses, elite_size=elite_size)
        fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in population])

        best_fitnesses.append(np.max(fitnesses))
        avg_fitnesses.append(np.mean(fitnesses))

        print(f"Generation {generation}: Best Fitness = {best_fitnesses[-1]}, Average Fitness = {avg_fitnesses[-1]}")
    return best_fitnesses, avg_fitnesses
