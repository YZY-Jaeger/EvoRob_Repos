from swarmy.perception import Perception
import pygame
import math
import numpy as np

class LightIntensitySensor_L(Perception):
    def __init__(self, agent, environment, config):
        super().__init__(agent, environment)
        self.agent = agent
        self.environment = environment
        self.config = config

    def sensor(self):
        """ Returns the light intensity at the point of the sensor.
        Light intensity is calculated as the average of the RGB values of the pixel.
        """

        robot_position_x, robot_position_y, robot_heading = self.agent.get_position()

        sensor_1_direction_x_l = math.sin(math.radians(robot_heading + 40))
        sensor_1_direction_y_l = math.cos(math.radians(robot_heading + 40))

        #trying to "install" the sensor to left front of robot
        light_l_pos = [(robot_position_x+(sensor_1_direction_x_l*35)),
                   (robot_position_y+(sensor_1_direction_y_l*35))]

        self.environment.add_dynamic_circle_object([(255,0,0), light_l_pos, 2, 1])
        # Get the surface of the environment
        surface = self.environment.displaySurface

        # Calculate the light intensity at the point of the sensor
        x, y = map(int, light_l_pos)
        #in case the sensor is out of the world, the following lines will make sure that the sensor is within the world
        x= x%self.config['world_width']
        y= y%self.config['world_height']

        #NOTE: the surface.get_at should use precise position of left sensor, here I use robot position to test
        r, g, b, _ = surface.get_at((x,y))


        light_intensity = (r + g + b) / (3*255)  # Average of RGB values
        print("x: ", int(robot_position_x), "y: ", int(robot_position_y))#print coordinates of the robot
        print("Light Intensity L: ", light_intensity)#print light intensity

        return light_intensity#to be used in controller