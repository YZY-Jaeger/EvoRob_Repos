import numpy as np
import random
from swarmy.actuation import Actuation  # Assuming this imports are correct
from evolution.evolution import Evolution  # Assuming this imports are correct
from copy import deepcopy

def activation_function(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

class ANN:
    def __init__(self, input_size=3, hidden_size=2, output_size=2):
        # Initialize weights and biases with small random values
        self.weights_input_hidden = np.random.uniform(-2, 2, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-2, 2, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-1, 1, hidden_size)
        self.bias_output = np.random.uniform(-1, 1, output_size)

    def forward(self, inputs):
        # Feedforward
        #hidden = activation_function(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        hidden = activation_function(np.dot(inputs, self.weights_input_hidden) )
        output = activation_function(np.dot(hidden, self.weights_hidden_output) + self.bias_output)
        return output

    def mutate(self, mutation_value=1):
        # Mutate weights and biases by adding a small random value
        self.weights_input_hidden += np.random.randn(*self.weights_input_hidden.shape) * mutation_value
        self.weights_hidden_output += np.random.randn(*self.weights_hidden_output.shape) * mutation_value
        self.bias_hidden += np.random.randn(*self.bias_hidden.shape) * mutation_value
        self.bias_output += np.random.randn(*self.bias_output.shape) * mutation_value

class EvolutionaryANNController(Actuation):
    def __init__(self, agent, config, population_size=10, elitism_size=2, mutation_rate=0.1):
        super().__init__(agent)
        self.config = config
        self.population_size = population_size
        self.elitism_size = elitism_size
        self.mutation_rate = mutation_rate
        self.population = [ANN(input_size=3, hidden_size=2, output_size=2) for _ in range(population_size)]
        self.generation = 0

    def evolve_population(self):
        # Calculate fitness for each ANN in the population
        fitnesses = [Evolution.calculate_fitness(self.agent) for ann in self.population]
        print(f"Fitnesses: {fitnesses}")
        sorted_fitnesses = sorted(fitnesses, reverse=True)   
        sum_fitnesses = sum(sorted_fitnesses)
        sorted_fitnesses = [fit / sum_fitnesses for fit in sorted_fitnesses]#normalize fitnesses

        sorted_pop = sorted(zip(self.population, fitnesses), key=lambda x: x[1], reverse=True)
        elites = [ind for ind, fit in sorted_pop[:self.elitism_size]]

        # Extract just the ANNs for selection
        sorted_anns = [ann for ann, _ in sorted_pop]

        # Generate new population with mutation and elitism
        new_population = elites[:]
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(sorted_anns, 2, replace=False, p = sorted_fitnesses)
            offspring = self.crossover(parent1, parent2)
            if random.random() < 0.8:
                self.mutate(offspring)
            new_population.append(offspring)

        self.population = new_population
        self.generation += 1
        print(f"Generation {self.generation} - Best fitness: {max(fitnesses)}")
        
    def crossover(self, parent1, parent2):
        child = deepcopy(parent1)
        for attr in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
            # Blend crossover
            child_weights = (parent1.__dict__[attr] + parent2.__dict__[attr]) / 2
            child.__dict__[attr] = child_weights
        return child

    def mutate(self, offspring):
        # Mutation for weights and biases
        for attr in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
            if np.random.rand() < self.mutation_rate:
                mutation_value = np.random.normal(0, 0.1, size=offspring.__dict__[attr].shape)
                offspring.__dict__[attr] += mutation_value

    def controller(self):
        # Each control step will evaluate the current population and evolve it
        self.evolve_population()

        # Use the best current ANN to control the robot
        best_ann = max(self.population, key=lambda ann: Evolution.calculate_fitness(self.agent))
        print(f"Best ANN: {best_ann.__dict__}")
        inputs = np.array(self.agent.get_perception()[1])
        vl, vr = best_ann.forward(inputs)
        max_speed = 3
        self.linear_velocity = min(abs((vl + vr) / 2), max_speed)
        self.angle_velocity = vr - vl

        self.update_position(self.linear_velocity, self.angle_velocity)

    def update_position(self, speed, turn_angle):
        # Update robot's position based on speed and turn angle
        x, y, heading = self.agent.get_position()
        dx = speed * np.sin(np.radians(heading))
        dy = speed * np.cos(np.radians(heading))
        new_x = x + dx
        new_y = y + dy
        new_heading = int((heading + turn_angle) % 360)
        self.agent.set_position(new_x, new_y, new_heading)
