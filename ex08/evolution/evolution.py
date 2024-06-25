import numpy as np
import math
import pygame
class Evolution():
    @staticmethod
    def is_colliding_with_wall(agent, static_objects):
        # Get the current position of the agent
        x, y, _ = agent.get_position()
        agent_rect = pygame.Rect(x, y, 1, 1)  # Assuming the agent can be represented as a 1x1 rectangle for collision detection

        # Check collision with each static rectangle object
        for obj in static_objects:
            _, rect, _ = obj  # Assuming the structure is ['COLOR', pygame.Rect(), border_width]
            if agent_rect.colliderect(rect):
                return True
        return False

    @staticmethod
    def calculate_fitness(agent, grid_size=5):
        # Check if the agent is colliding with a wall
        if Evolution.is_colliding_with_wall(agent, agent.environment.staticRectList):
            #fitness -= 0.0001 #punish the agent for colliding with a wall
            return -1

        # Get the current position of the agent
        x, y, _ = agent.get_position()

        # Check if the agent is within the 500x500 grid
        if not (20 <= x <= agent.config['world_width']-20 and 20 <= y <= agent.config['world_height']-20):
            return -1
        
        # Calculate the grid cell that the agent is currently in
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)
        # Add the current grid cell to the set of visited cells
        agent.visited_grid_cells.add((grid_x, grid_y))

        # The fitness is the number of unique grid cells visited / total number of grid cells
        fitness = len(agent.visited_grid_cells) / (agent.config['world_width']/grid_size * agent.config['world_height']/grid_size)

        return fitness
    

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