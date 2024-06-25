import numpy as np
import random
from swarmy.actuation import Actuation  # Assuming this imports are correct
from evolution.evolution import Evolution  ,ANN# Assuming this imports are correct
from copy import deepcopy

def activation_function(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

class EvolutionaryANNController(Actuation):
    def __init__(self, agent, config, population_size=10, elitism_size=2, mutation_rate=0.1):
        super().__init__(agent)
        self.config = config
        self.population_size = population_size
        self.elitism_size = elitism_size
        self.mutation_rate = mutation_rate
        self.ann = ANN()
        self.generation = 0

    def evolve_population(self):
        # Calculate fitness for each ANN in the population
        fitnesses = [Evolution.calculate_fitness(self.agent) for ann in self.population]
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
        #print(f"Generation {self.generation} - Best fitness: {max(fitnesses)}")
        
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
        #self.evolve_population()

        # Use the best current ANN to control the robot
        #best_ann = max(self.population, key=lambda ann: Evolution.calculate_fitness(self.agent))
        inputs = np.array(self.agent.get_perception()[1])
        vl, vr = self.ann.forward(inputs)
        max_speed = 3
        self.linear_velocity = min(abs((vl + vr) / 2), max_speed)
        self.angle_velocity = vr - vl
        Evolution.calculate_fitness(self.agent) # Update fitness for the current ANN
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
