from swarmy.agent import Agent
import random
import pygame
import numpy as np
class MyAgent(Agent):
    def __init__(self, environment, controller, sensor, config):
        super().__init__(environment, controller, sensor, config)

        self.environment = environment
        self.trajectory = []



    def initial_position(self):
        """
        Define the initial position of the agent.
        Hint:
        Use x,y,gamma = self.set_position(x-position, y-position, heading) to set the position of the agent.
        """
        x = random.randint(0, self.config['world_width'])
        y = random.randint(0, self.config['world_height'])

        gamma = random.randint(0, 360)
        self.set_position(x, y, gamma)

    def save_information(self, last_robot):
        """
        Save information of the agent, e.g. trajectory or the environmental plot.
        Hint:
        - Use pygame.draw.lines() to draw the trajectory of the robot and access the surface of the environment with self.environment.displaySurface
        - pygame allows to save an image of the current environment
        """
        print("Save information not implemented, check my_agent.py")
        """ your implementation here """

        pass



    def update_position(self, speed, turn_angle):
        """
        Update the agent's position and heading over time.
        """
        # Limit the speed and turn angle to their maximum values
        speed = min(speed, self.config['max_speed'])
        turn_angle = min(turn_angle, self.config['max_turn_angle'])

        # Calculate the change in position
        dx = speed * np.cos(np.radians(self.actuation.angle))
        dy = speed * np.sin(np.radians(self.actuation.angle))

        # Update the heading
        self.actuation.angle += turn_angle
        self.actuation.angle %= 360  # Ensure the heading is between 0 and 360

        # Update the position
        new_x = (self.actuation.position[0] + dx) % self.config['world_width']
        new_y = (self.actuation.position[1] + dy) % self.config['world_height']
        self.set_position(new_x, new_y, self.actuation.angle)
            
        #print(f"Updated position: ({new_x}, {new_y}), heading: {self.actuation.angle}")







