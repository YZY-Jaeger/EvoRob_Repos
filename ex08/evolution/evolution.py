import numpy as np
import math

class Evolution():
    @staticmethod
    def calculate_fitness(agent, grid_size=5):
        # Get the current position of the agent
        x, y, _ = agent.get_position()

        # Check if the agent is within the 500x500 grid
        if not (20 <= x <= agent.config['world_width']-20 and 20 <= y <= agent.config['world_height']-20):
            return -1
        
        # Calculate the grid cell that the agent is currently in
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)
        #print(grid_x,grid_y)
        # Add the current grid cell to the set of visited cells
        agent.visited_grid_cells.add((grid_x, grid_y))

        # The fitness is the number of unique grid cells visited / total number of grid cells
        fitness = len(agent.visited_grid_cells)/(agent.config['world_width']/grid_size * agent.config['world_height']/grid_size)

        return fitness