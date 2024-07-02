import numpy as np
import random

class RingProblem:
    def __init__(self):
        self.num_cells = 40  # Total number of cells on the ring
        self.cell_indices = list(range(self.num_cells))
        
        # Randomly assign 20 cells to the left half and the remaining to the right half
        self.left_half_indices = random.sample(self.cell_indices, self.num_cells // 2)
        self.right_half_indices = [i for i in self.cell_indices if i not in self.left_half_indices] 
        
        # Initialize the agent's position randomly within the ring
        self.agent_position = random.randint(0, self.num_cells - 1)
        
        # Initialize the weights of the neural network with random values
        self.network_weights = np.random.rand(20)
    
    def get_input_vector(self, current_cell_number):
        # Ensure the cell_number is within the range of 0 to 19
        normalized_cell_number = current_cell_number % 20

        # Creates a vector of 20 zeros
        input_vector = np.zeros(20)
        
        # Set the position corresponding to the cell number to 1
        input_vector[normalized_cell_number] = 1

        return input_vector

    def activation_function(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def move_agent(self):
        # Get the current cell number where the agent is positioned
        current_cell_number = self.agent_position
        
        # Convert the cell number to a one-hot encoded input vector
        input_vector = self.get_input_vector(current_cell_number)
        
        # Compute the net input to the neural network by dot product : multiplied with the corresponding weight
        net_input = np.dot(input_vector, self.network_weights)
        
        # Get the output from the activation function (sigmoid)
        output = self.activation_function(net_input)
        
        # Move the agent based on the output: CW if output >= 0.5, otherwise CCW
        if output > 0.5:
            self.agent_position = (self.agent_position + 1) % self.num_cells  # Move Clockwise
        else:
            self.agent_position = (self.agent_position - 1) % self.num_cells  # Move Counterclockwise

    def evaluate_fitness(self, steps=1000):
        left_half_time = 0
        
        # Simulate the agent's movements for a specified number of steps
        for _ in range(steps):
            self.move_agent()
            # Check if the agent is in the left half of the ring
            if self.agent_position in self.left_half_indices:
                left_half_time += 1
        
        # Calculate the fitness score as the proportion of time spent in the left half
        return left_half_time / steps 
    
    def evolve(self, generations=20, population_size=20, mutation_rate=0.01):
        # Initialize a population of neural networks with random weights
        population = [np.random.rand(20) for _ in range(population_size)]
        fitness_history = []
        
        for generation in range(generations):
            fitness_scores = []
            
            # Evaluate fitness for each individual in the population
            for individual in population:
                self.network_weights = individual
                fitness = self.evaluate_fitness()
                fitness_scores.append((fitness, individual))
                
                # Check if fitness has reached more than 0.998
                if fitness > 0.998:
                    print(f"Stopping evolution at Generation {generation}, Best Fitness: {fitness}")
                    return fitness_history
            
            # Sort the population by fitness scores in descending order
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            fitness_history.append(fitness_scores[0][0])

            # Keep the top four individuals for the next generation (elitism)
            new_population = [fitness_scores[i][1] for i in range(4)]
            
            # Fill the rest of the new population using crossover and mutation
            while len(new_population) < population_size:
                # Select two parents from the top 10% of the population
                parent1, parent2 = random.choices(fitness_scores[:population_size // 10], k=2)
                # Create a child by combining the parents' weights
                child = self.crossover(parent1[1], parent2[1])
                # Mutate the child's weights
                child = self.mutate(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
            
            print(f"Generation {generation}, Best Fitness: {fitness_scores[0][0]}")
        
        # Set the network weights to the best individual from the final generation
        self.network_weights = fitness_scores[0][1]
        return fitness_history
    
    def crossover(self, parent1, parent2):
        # Perform single-point crossover between two parents
        crossover_point = random.randint(0, 19)
        return np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    
    def mutate(self, individual, rate):
        # Mutate the individual's weights with a given mutation rate
        for i in range(len(individual)):
            if random.random() < rate:
                individual[i] += np.random.normal()
        return individual
    
    def get_motion_directions(self):
        directions = []

        # Iterate through each cell in the ring
        for cell_number in range(40):
            # Set the agent's position to the current cell
            self.agent_position = cell_number

            # Move the agent according to the neural network's output
            self.move_agent()

            # Determine and record the direction of the movement
            if self.agent_position == (cell_number + 1) % 40:
                directions.append("CW")
            else:
                directions.append("CCW")
        return directions

# Run the simulation
ring_problem = RingProblem()
fitness_history = ring_problem.evolve()

# Output motion directions
motion_directions = ring_problem.get_motion_directions()
print("Motion Directions for each cell:", motion_directions)
