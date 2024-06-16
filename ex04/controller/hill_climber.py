import random
from swarmy.actuation import Actuation
import yaml
import math
from evolution.evolution import Evolution
class HillClimber(Actuation):
    def __init__(self, agent, config):
        super().__init__(agent)
        self.config = config
        self.init_pos = True
        self.control_params = [random.uniform(-1, 1) for _ in range(6)]  # Initialize a random genome
        self.best_fitness = 0  # Initialize the best fitness to a very low value

    def controller(self):
        if self.init_pos:
            self.agent.initial_position()
            self.init_pos = False

        max_speed = 3
        max_angle = 50
        # Get sensor data
        sensor_id, sensor_values = self.agent.get_perception()
        #sensor_l, sensor_r, sensor_m = sensor_values
        sensor_l, sensor_r, sensor_m = [value / 3 for value in sensor_values]
        # Calculate wheel speeds based on the control parameters
        vl = self.control_params[0]*sensor_l + self.control_params[1]
        vr = self.control_params[2]*sensor_r + self.control_params[3] + self.control_params[4]*sensor_m + self.control_params[5]

        # Calculate new linear and angular velocities
        #self.linear_velocity = min(abs((vl + vr) / 2),max_speed)#take the smaller speed
        #self.angle_velocity = (vr - vl) /5

        self.linear_velocity = min(abs((vl + vr) / 2),max_speed)#take the smaller speed
        self.angle_velocity = (vr - vl) 
        # Update the position
        self.update_position(self.linear_velocity, self.angle_velocity)

        # Calculate the fitness
        fitness = Evolution.calculate_fitness(self.agent)
        #print("Fitness: ",fitness)
        # If the fitness has improved, keep the control parameters. Otherwise, revert to the previous control parameters.
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.agent.current_generation +=1
            print("Fitness: ",fitness)
            print("Generation: ", self.agent.current_generation)
        else:
            self.control_params = self.previous_control_params


        # Mutate the control parameters to create a new candidate
        self.previous_control_params = self.control_params
        #range is -1 to 1
        #print( self.control_params)
        self.control_params = [min(max(param + random.uniform(-0.2, 0.2), -1), 1) for param in self.control_params]

    def update_position(self, speed, turn_angle):
        # Get current position and heading
        x, y, heading = self.agent.get_position()

        # Calculate the change in position
        dx = speed * math.sin(math.radians(heading))
        dy = speed * math.cos(math.radians(heading))
        # Calculate new position
        new_x = (x + dx) 
        new_y = (y + dy)
        # Calculate new heading
        #new_heading = round((heading + turn_angle) % (2 * math.pi))
        new_heading = int(     (heading + turn_angle) % 360 )
        # Set new position and heading
        self.agent.trajectory.append((new_x, new_y))
        # Save the image
        self.agent.save_information(None)
        self.agent.set_position(new_x, new_y, new_heading)

    def torus(self):
        # Get current robot position and heading
        robot_position_x, robot_position_y, robot_heading = self.agent.get_position()

        # Implement torus world by manipulating the robot position here

        # Update robot's position and heading
        self.agent.set_position(robot_position_x, robot_position_y, robot_heading)