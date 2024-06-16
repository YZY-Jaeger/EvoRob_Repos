from swarmy.environment import Environment
import pygame
import numpy as np
import random

class My_environment(Environment):
    def __init__(self, config):
        self.config = config
        super().__init__(config)

        self.light_dist_surface = self.calculate_light_surface()

        self.light_dist = self.defineLight()

    def calculate_light_surface(self):#calculate it once
            light_dist = self.defineLight()
            light_dist_normalized = (light_dist / light_dist.max() * 255).astype(np.uint8)  # Normalize to range 0-255
            return pygame.surfarray.make_surface(np.stack([light_dist_normalized]*3, axis=-1))  # Create a surface from the light distribution
        


    def add_static_rectangle_object(self):
        """
        Add static rectangle object to the environment such as walls or obstacles.
        Example:
            self.staticRectList.append(color, pygame.Rect(x, y, width, height), border_width))
        Returns:
        """

        wall_thickness = 10  # Increase this value to make the walls thicker
        self.staticRectList.append(['BLACK', pygame.Rect(5, 5, self.config['world_width'] - wall_thickness, wall_thickness), wall_thickness])
        self.staticRectList.append(['BLACK', pygame.Rect(5, 5, wall_thickness, self.config['world_height'] - wall_thickness), wall_thickness])
        self.staticRectList.append(['BLACK', pygame.Rect(5, self.config['world_height'] - wall_thickness, self.config['world_width'] - wall_thickness, wall_thickness), wall_thickness])
        self.staticRectList.append(['BLACK', pygame.Rect(self.config['world_width'] - wall_thickness, 5, wall_thickness, self.config['world_height'] - wall_thickness), wall_thickness])

    def add_random_walls(self):

        color = (0, 255, 255)  # Wall color
        wall_thickness = 10  # Line thickness

            # Random start point within the bounds
        for _ in range(random.randint(4,10)): 
            #vertical wall
            if random.choice([True, False]): # 50% chance for True or False
                #vertical top
                left=random.randint(5 + wall_thickness, self.config['world_width']- 2* wall_thickness)
                top = 5 + wall_thickness
                height = random.randint(50, 200)
                self.staticRectList.append([color, pygame.Rect(left, top, wall_thickness, height), wall_thickness])
            else:
                #vertical bottom
                left=random.randint(5 + wall_thickness, self.config['world_width']- 2*wall_thickness)
                top = self.config['world_width'] - wall_thickness
                height = random.randint(50, 200)
                self.staticRectList.append([color, pygame.Rect(left, top - height, wall_thickness, height), wall_thickness])   
        
        for _ in range(random.randint(4,10)):
            #horizatal wall
            if random.choice([True, False]):
                #horizontal left
                left = 5 + wall_thickness
                top = random.randint(wall_thickness, self.config['world_width']-wall_thickness)
                width = random.randint(50, 200)
                self.staticRectList.append([color, pygame.Rect(left, top, width, wall_thickness), wall_thickness])

            else:
                #horizontal right
                left = self.config['world_width'] - wall_thickness
                top = random.randint(wall_thickness, self.config['world_width']-wall_thickness)
                width = random.randint(50, 200)
                self.staticRectList.append([color, pygame.Rect(left - width, top,width, wall_thickness), wall_thickness])



            """
            # Determine whether to adjust end_x or end_y
            if random.choice([True, False]):  # 50% chance for True or False
                start_x = random.randint(0, self.config['world_width'])
                # Adjust end_x, keep end_y the same as start_y
                end_x = start_x+random.randint(50, 300)
                left=start_x
                top=end_x
                self.staticRectList.append([color, pygame.Rect(left, top, wall_thickness, top - wall_thickness,), wall_thickness])
            else:
                # Adjust end_y, keep end_x the same as start_x
                start_y = random.randint(0, self.config['world_height'])
                end_y = start_y+random.randint(50, 300)
                left=start_y
                top=end_y
                self.staticRectList.append([color, pygame.Rect(left, top, left - wall_thickness, wall_thickness), wall_thickness])
            """
            
    




    def add_static_circle_object(self):
        """
        Add static circle object to the environment such as sources or sinks.
        Example:
            self.staticCircList.append([color, position, border_width, radius])
        Returns:
        """
        pass


    def set_background_color(self):
        """
        Set the background color of the environment.
        Example:
            self.displaySurface.fill(self.BACKGROUND_COLOR)
        Hint: It is possible to use the light distribution to set the background color.
        For displaying a light distribution you might find pygame.surfarray.make_surface and self.displaySurface.blit useful)
        Returns:
        """
        #self.displaySurface.blit(self.light_dist_surface, (0, 0))  # Draw the precalculated light distribution on the display surface
        self.displaySurface.fill(self.BACKGROUND_COLOR)
    ###  LIGHT DISTRIBUTION ###

    def defineLight(self):
        """
        Define the light distribution of the environment.
        Returns: 3 dimensional light distribution tuple (x,y,light_intensity)
        """

        
        light_source_position = np.array([self.config['world_width'] / 2, self.config['world_height'] / 2])
        max_light_intensity = 60  # Maximum light intensity at the source

        light_dist = np.zeros((self.config['world_width'], self.config['world_height']))

        for x in range(self.config['world_width']):
            for y in range(self.config['world_height']):
                distance = np.linalg.norm(np.array([x, y]) - light_source_position)
                light_dist[x, y] = max(0, max_light_intensity - distance/3)

        return light_dist
       
