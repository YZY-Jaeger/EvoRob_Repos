import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sigmoid activation function with negative outputs allowed
def sigmoid(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

import numpy as np

def sigmoid(x):
    # Using a sigmoid activation function that allows for negative outputs
    return 2 / (1 + np.exp(-2 * x)) - 1

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=2, output_size=1,
                 weights=None, hidden_weights=None, biases=None, hidden_biases=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights from input to hidden layer
        if weights is None:
            self.weights = 2*np.random.rand(self.input_size, self.hidden_size)-1  # 2 input neurons, 2 hidden neurons
        else:
            self.weights = weights
        
        # Initialize weights from hidden to output layer
        if hidden_weights is None:
            self.hidden_weights = 2 * np.random.rand(self.hidden_size, self.output_size) - 1   # 2 hidden neurons, 1 output neuron
        else:
            self.hidden_weights = hidden_weights

        # Initialize biases for hidden layer
        if biases is None:
            self.biases =  np.random.rand(self.hidden_size)  # 1 bias for each hidden neuron
        else:
            self.biases = biases



    def get_weights(self):
        # Concatenate all weights and biases into one flat array
        return np.concatenate((self.weights.flatten(), self.hidden_weights.flatten(),
                               self.biases.flatten()))

    def set_weights(self, weights):
        # The number of weights in the first layer
        num_input_to_hidden_weights = self.weights.size
        # The number of weights in the hidden to output layer
        num_hidden_to_output_weights = self.hidden_weights.size

        # Split the flat weight array back into the original structure
        self.weights = weights[:num_input_to_hidden_weights].reshape(self.input_size, self.hidden_size)
        offset = num_input_to_hidden_weights
        self.hidden_weights = weights[offset:offset + num_hidden_to_output_weights].reshape(self.hidden_size, self.output_size)
        offset += num_hidden_to_output_weights
        self.biases = weights[offset:offset + self.biases.size].reshape(self.hidden_size)
        offset += self.biases.size


    def forward(self, inputs):
        # Add bias term to inputs and pass through the first layer
        hidden_inputs = np.dot(inputs, self.weights) 
        hidden_outputs = sigmoid(hidden_inputs)

        # Pass output of the first layer to the output layer
        final_inputs = np.dot(hidden_outputs, self.hidden_weights) 
        final_output = sigmoid(final_inputs) - self.biases  # Subtract bias 

        return final_output[0]  # Ensure it returns a scalar if final_output is an array



def evaluate_fitness(network):
    total_error = 0
    inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    expected_outputs = np.array([0, 1, 1, 0])  # XOR truth table

    for i in range(len(inputs)):
        output = network.forward(inputs[i])
        # Ensure output is correctly dimensioned and comparison is valid
        total_error += 1 - np.abs(expected_outputs[i] - output)
    fitness = total_error / 4  # Normalize total error
    return fitness

def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        network = NeuralNetwork()  # This uses default sizes as defined in the NeuralNetwork class constructor
        population.append(network)
    return population





def rank_based_selection(population,fitness_scores):
    # Step 1: Evaluate fitness and sort the population
    #fitness_scores = [evaluate_fitness(network) for network in population]
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]


    # Step 2: Divide sorted population into 8 ranks
    num_ranks = 8
    population_size = len(sorted_population)
    rank_size = population_size // num_ranks
    ranks = np.zeros(population_size)
    
    for rank in range(num_ranks):
        start_index = rank * rank_size
        # For the last rank, include all remaining individuals
        end_index = start_index + rank_size if rank < num_ranks - 1 else population_size
        ranks[start_index:end_index] = rank + 1  # Assign rank (1 to 8)

   
    probabilities = np.array([1 / rank for rank in ranks])
    
    # Step 4: Normalize probabilities
    probabilities /= np.sum(probabilities)
    # Step 5: Select parents based on probabilities
    parent1_idx, parent2_idx = np.random.choice(population_size, 2, replace=False, p=probabilities)
    
    return sorted_population[parent1_idx], sorted_population[parent2_idx]

def crossover(parent1, parent2):
    parent1_weights = parent1.get_weights()
    parent2_weights = parent2.get_weights()
    crossover_point = np.random.randint(0, len(parent1_weights))
    child_weights = np.concatenate((parent1_weights[:crossover_point], parent2_weights[crossover_point:]))
    return child_weights

def mutate(weights, mutation_rate):
    mutation = np.random.uniform(-mutation_rate, mutation_rate, size=weights.shape)
    return weights + mutation

def evolve(population, mutation_rate):
    elitism_count = int(0.10 * len(population))  # elistism: keep top 5%
    sorted_population = sorted(population, key=lambda network: evaluate_fitness(network), reverse=True)
    fitness_scores = [evaluate_fitness(network) for network in sorted_population]

    new_population = sorted_population[:elitism_count]  # Elitism: Keep top 5%
    
    while len(new_population) < len(population):
        parent1, parent2 = rank_based_selection(sorted_population,fitness_scores)
        child_weights = crossover(parent1, parent2)
        child_weights = mutate(child_weights, mutation_rate)
        child = NeuralNetwork(2, 2, 1)
        child.set_weights(child_weights)
        new_population.append(child)
    return new_population, fitness_scores

# Main evolutionary loop
population_size = 1000
mutation_rate = 0.3
input_size = 2
hidden_size = 2
output_size = 1
max_generations = 2000

# Initialize the population correctly
population = initialize_population(population_size)
generation = 0
best_fitness = 0
best_network = None

print("Starting evolution...")
while generation < max_generations and best_fitness < 0.99:
    
    population, fit = evolve(population, mutation_rate)
    
    # Update the best fitness and best network if found
    current_best_fitness = max(fit)
    if current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
        best_network = population[np.argmax(fit)]
    best_fitness = max(fit)


    if generation % 10 == 0:
        print(f"Generation {generation}: Best fitness = {best_fitness}")
    generation += 1

print("Evolution completed.")
print(f"Best fitness achieved: {best_fitness}")

# 3D plot of the output
resolution = 50
x = np.linspace(0, 1, resolution)
y = np.linspace(0, 1, resolution)
X, Y = np.meshgrid(x, y)
Z = np.array([best_network.forward([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Network Output')
ax.set_title('3D Visualization of XOR Neural Network Output')
fig.colorbar(surf)
plt.show()
