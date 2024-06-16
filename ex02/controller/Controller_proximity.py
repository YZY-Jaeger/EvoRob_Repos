from swarmy.perception import Perception
from swarmy.actuation import Actuation
import pygame
import math
import numpy as np


class Proximity_Controller(Actuation):
   
    def __init__(self, agent, config):
        super().__init__(agent)
        self.config = config
        self.init_pos = True

    def controller(self):
        if self.init_pos:
            self.agent.initial_position()
            self.init_pos = False

        # Get sensor data
        perception = self.agent.get_perception()
        sensor_id, sensor_values = perception
        sensor_l, sensor_r, sensor_m = sensor_values

        # Define a threshold distance to detect objects
        threshold_distance = 50
        base_speed = 5
        turn_angle = 0
        print(f"L: {sensor_l}, M: {sensor_m}, R: {sensor_r}")

        if sensor_l == 0 and sensor_r == 0 and sensor_m == 0:
            speed  = base_speed
        elif sensor_m != 0 and sensor_l != 0: 
            speed = base_speed
            turn_angle = -10  # turn right
        elif sensor_m != 0 and sensor_r != 0: 
            speed = base_speed
            turn_angle = 10  # turn left
        else:
            speed = 1

        self.update_position(speed, turn_angle)

    def update_position(self, speed, turn_angle):
        """
        Update the agent's position and heading over time.
        """
        # Limit the speed and turn angle to their maximum values
        max_speed = 1  # maximum distance the robot can cover in one time step
        max_turn_angle = 45  # maximum angle the robot can turn per time step
        speed = min(speed, max_speed)
        turn_angle = min(turn_angle, max_turn_angle)
       
        # Get the current position and heading
        x, y, gamma = self.agent.get_position()

        # Calculate the change in position
        dx = speed * math.sin(math.radians(gamma))
        dy = speed * math.cos(math.radians(gamma))

        # Update the heading
        new_heading = (gamma + turn_angle) % 360

        # Update the position 
        new_x = (x + dx) % self.config['world_width']
        new_y = (y + dy) % self.config['world_height']
        self.agent.set_position(new_x, new_y, new_heading)
        current_time = pygame.time.get_ticks()  # Get the current time in milliseconds
        self.agent.trajectory.append((new_x, new_y, current_time))
        # Save the image
        self.agent.save_information(None)

    def torus(self):
        """
        Implement torus world by manipulating the robot position. Again self.agent.get_position and self.agent.set_position might be useful
        """
        robot_position_x, robot_position_y, robot_heading = self.agent.get_position()

        # Implement torus world by manipulating the robot position, here.

        robot_position_x = robot_position_x % self.config['world_width']
        robot_position_y = robot_position_y % self.config['world_height']

        self.agent.set_position(robot_position_x, robot_position_y, robot_heading)