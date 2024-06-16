import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sigmoid activation function with negative outputs allowed
def sigmoid(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

# Define the Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.hidden_size, self.input_size + 1))  # +1 for bias
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.output_size, self.hidden_size + 1))  # +1 for bias
    
    # Forward propagation through the network
    def forward(self, inputs):
        # Add bias to inputs
        inputs_bias = np.append(inputs, 1)  # bias neuron
        
        # Hidden layer activation
        hidden_inputs = np.dot(self.weights_input_hidden, inputs_bias)
        hidden_outputs = sigmoid(hidden_inputs)
        
        # Output layer activation
        output_inputs = np.dot(self.weights_hidden_output, np.append(hidden_outputs, 1))  # bias neuron
        final_output = sigmoid(output_inputs)
        
        return final_output
    
    # Get and set weights for the genetic algorithm
    def get_weights(self):
        return np.concatenate((self.weights_input_hidden.flatten(), self.weights_hidden_output.flatten()))
    
    def set_weights(self, weights):
        split_idx = self.hidden_size * (self.input_size + 1)
        self.weights_input_hidden = weights[:split_idx].reshape(self.hidden_size, self.input_size + 1)
        self.weights_hidden_output = weights[split_idx:].reshape(self.output_size, self.hidden_size + 1)

# Fitness function evaluation
def evaluate_fitness(network):
    total_error = 0
    inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    expected_outputs = np.array([0, 1, 1, 0])  # XOR truth table
    
    for i in range(len(inputs)):
        output = network.forward(inputs[i])
        total_error += 1 - np.abs(expected_outputs[i] - output)
    
    fitness = total_error / len(inputs)
    return fitness

# Genetic Algorithm functions
def initialize_population(population_size, input_size, hidden_size, output_size):
    population = []
    for _ in range(population_size):
        network = NeuralNetwork(input_size, hidden_size, output_size)
        population.append(network)
    return population

def rank_based_selection(population):
    sorted_population = sorted(population, key=lambda network: evaluate_fitness(network))
    ranks = np.arange(1, len(population) + 1)
    probabilities = [1 - (rank / np.sum(ranks)) for rank in ranks]
    probabilities /= np.sum(probabilities)
    parent1_idx, parent2_idx = np.random.choice(len(population), 2, replace=False, p=probabilities)
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
    elitism_count = int(0.05 * len(population))  # Keep top 5%
    sorted_population = sorted(population, key=lambda network: evaluate_fitness(network), reverse=True)
    new_population = sorted_population[:elitism_count]  # Elitism: Keep top 5%
    
    while len(new_population) < len(population):
        parent1, parent2 = rank_based_selection(sorted_population)
        child_weights = crossover(parent1, parent2)
        child_weights = mutate(child_weights, mutation_rate)
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
        child.set_weights(child_weights)
        new_population.append(child)
    return new_population

# Main evolutionary loop
population_size = 10
mutation_rate = 0.1
input_size = 2
hidden_size = 2
output_size = 1
max_generations = 200

population = initialize_population(population_size, input_size, hidden_size, output_size)
generation = 0
best_fitness = 0
best_network = None

print("Starting evolution...")
while generation < max_generations and best_fitness < 0.99:
    print(f"Generation {generation}...")
    population = evolve(population, mutation_rate)
    best_network = max(population, key=lambda network: evaluate_fitness(network))
    best_fitness = evaluate_fitness(best_network)
    if generation % 100 == 0:
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
