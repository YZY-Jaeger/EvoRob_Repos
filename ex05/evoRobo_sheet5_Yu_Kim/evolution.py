import numpy as np
from functions import fitness

# Initialize the population with random values within given bounds
def initialize_population(size):
    return np.random.uniform(-32.768, 32.768, (size, 3))#here we just generate random values for x,y,z as Genetic Representation

def tournament_selection(population, fitnesses, num_parents, tournament_size=3):#tournament_size is default 3, we divide the population into groups of 3 and select the best one
    selected = []
    for _ in range(num_parents):
        # Randomly pick several candidates from the population
        participants = np.random.choice(len(population), tournament_size, replace=False)
        # Choose the best candidate based on fitness
        best_participant = participants[np.argmax(fitnesses[participants])]
        selected.append(population[best_participant])
    return np.array(selected)

# Perform simulated binary crossover (SBX) between two parents
def sbx_crossover(parent1, parent2, eta=1.0, prob_crossover=0.9):
    children1, children2 = parent1.copy(), parent2.copy()
    # Apply crossover with a certain probability
    if np.random.rand() < prob_crossover:
        for i in range(len(parent1)):
            # Random chance to swap each gene
            if np.random.rand() < 0.5:
                # Compute a blending factor beta
                beta = np.random.uniform(-0.1, 0.1)
                # Blend genes of parents to produce children
                children1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                children2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                # Ensure children's genes are within the specified limits
                children1[i] = np.clip(children1[i], -32.768, 32.768)
                children2[i] = np.clip(children2[i], -32.768, 32.768)
    return children1, children2


# Mutate an individual by altering its genes slightly
def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.1):#sigma is the offset value for mutation, default is 0.1
    for i in range(len(individual)):
        # Apply mutation to each gene with a certain probability
        if np.random.rand() < mutation_rate:#mutation_rate is default 0.1, we apply mutation to 10% of the genes
            individual[i] += np.random.normal(0, sigma)
            individual[i] = np.clip(individual[i], -32.768, 32.768)
        
    return individual


# Replace the old population with the new one while maintaining some of the best
def elitist_replacement(old_population, offspring, old_fitnesses, offspring_fitnesses, elite_size=5):#we dont use fixed elite_size for now
    # Combine old population and offspring into one pool
    combined_population = np.vstack((old_population, offspring))
    combined_fitnesses = np.concatenate((old_fitnesses, offspring_fitnesses))
    # Select the best individuals based on fitness
    indices = np.argsort(-combined_fitnesses)#we sort the fitnesses array in descending order
    new_population = combined_population[indices[:len(old_population)]]
    
    print("New Population after Replacement:\n", new_population)  # Debug statement
    return new_population #as we dont have fixed elite_size, we return the same size of old_population
                                                            #for example if we have size 100 of combined_population, we return the BEST 50(same as old population) individuals



def evolutionary_algorithm(pop_size, num_generations):
    # Initialize the population
    population = initialize_population(pop_size)
    fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in population])

    best_fitnesses = []
    avg_fitnesses = []

    for generation in range(num_generations):
        # Evaluate the fitness of each individual
        fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in population])
        # Select parents for reproduction
        parents = tournament_selection(population, fitnesses, pop_size )#third parameter is number of parents to be selected, default we let every individual to be parent
        offspring = []
        # Make sure we have an even number of parents
        num_parents = len(parents) - 1  # Ensure we have an even number, ignore last if odd
        for i in range(0, num_parents, 2):
            # Crossover parents to create children
            child1, child2 = sbx_crossover(parents[i], parents[i+1])
            offspring.append(child1)
            offspring.append(child2)
        offspring = np.array(offspring)
        # Mutate the offspring
        mutated_offspring = np.array([gaussian_mutation(ind) for ind in offspring])
        # Recalculate fitness for the new offspring
        offspring_fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in mutated_offspring])

        # Replace the old population with the new one
        population = elitist_replacement(population, mutated_offspring, fitnesses, offspring_fitnesses)
        fitnesses = np.array([fitness(ind[0], ind[1], ind[2]) for ind in population])
    
        # Find and print the best fitness in the current population
        best_fitnesses.append(np.max(fitnesses))
        avg_fitnesses.append(np.mean(fitnesses))

        print(f"Generation {generation}: Best Fitness = {best_fitnesses[-1]}, Average Fitness = {avg_fitnesses[-1]}")
        print("Fitnesses:\n", fitnesses)  # Debug statement
    return best_fitnesses, avg_fitnesses