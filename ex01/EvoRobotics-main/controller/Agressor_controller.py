import random

from swarmy.actuation import Actuation
from swarmy.perception import Perception
from sensors.light_sensor_L import LightIntensitySensor_L
from sensors.light_sensor_R import LightIntensitySensor_R
import yaml
import numpy as np
import math
from swarmy.environment import Environment

class Agressor_controller(Actuation):

    def __init__(self,agent, config):

        # rest of your code
        super().__init__(agent)
        """
        self.linear_velocity = <your value> # set the linear velocity of the robot
        self.angle_velocity =  <your value> # set the angular velocity of the robot
        """
        self.linear_velocity=5
        self.angle_velocity=5
        self.config = config
        self.init_pos = True            # flag to set initial position of the robot
        # Initialize sensors



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

        # Print out the values for debugging
        print(f"x: {x}, y: {y}, gamma: {gamma}, dx: {dx}, dy: {dy}, new_x: {new_x}, new_y: {new_y}, new_heading: {new_heading}")

    def controller(self):
        """
        This function overwrites the abstract method of the robot controller.
        these function might help:
        - self.stepBackward(velocity)
        - self.turn_right(angle_velocity)
        - self.turn_left(angle_velocity)
        - x,y,gamme = self.agent.get_position() # returns the current position and heading of the robot.
        - self.agent.set_position(new_position_x, new_position_y, robot_heading) # set the new position of the robot
        - self.agent.get_perception() returns the ID of the robot and sensor values that you implemented in sensor() of the class MySensor()
        Returns:
        """

        #Set initial robot position and direction
        if self.init_pos:
            self.agent.initial_position()
            self.init_pos = False

        sensor = self.agent.get_perception()
        delta_s = abs(sensor[1] - sensor[2])

        # Define a base speed and a sensitivity factor for how quickly the robot responds to light differences
        base_speed = 5
        sensitivity = 50  # Adjust this to change how reactive the robot is to light differences

        # Calculate speed based on light intensity difference
        speed = base_speed + sensitivity * delta_s

        # Determine the turning direction based on which sensor detects more light
        if sensor[1] > sensor[2]:
            turn_angle = 5  # Turn left
        elif sensor[1] < sensor[2]:
            turn_angle = -5  # Turn right
        else:
            turn_angle = 0  # No turn if both sensors read the same


        #delta_s = abs(light_intensity_L - light_intensity_R)
        '''
        if light_intensity_L > light_intensity_R:  # Adjust threshold based on expected light intensity differences
            speed = 5  # Example proportional control
            turn_angle = -10 #turn left
        
        elif light_intensity_L < light_intensity_R:
            speed = 5  # Example proportional control
            turn_angle = 10
        else: 
            speed=0
            turn_angle=0
        '''
        self.update_position(speed, turn_angle)


    def torus(self):
        """
        Implement torus world by manipulating the robot position. Again self.agent.get_position and self.agent.set_position might be useful
        """
        robot_position_x, robot_position_y, robot_heading = self.agent.get_position()

        # Implement torus world by manipulating the robot position, here.
        robot_position_x = robot_position_x % self.config['world_width']
        robot_position_y = robot_position_y % self.config['world_height']

        self.agent.set_position(robot_position_x, robot_position_y, robot_heading)


