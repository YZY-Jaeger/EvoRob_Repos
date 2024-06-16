# EvoRobotics Group Assignment

## Team Members
- **Minsol Kim**
- **Yu Zeyuan**

## Overview
This repository hosts our group's submission for Task Sheet 1 of the Evolutionary Robotics course. The submission includes all necessary code, data visualizations (plots), and supplementary resources essential for evaluating our implementation of specified robotic behaviors and sensor integrations.

## Tasks and Subtasks
### Completed Tasks

#### Task 1.1: Prepare the Robot Simulator for Braitenberg Vehicles
  - **Subtask 1.1**: Implemented a torus space in the controller.
    - Modified both `Agressor_controller.py` and `Fear_controller.py` to support toroidal wrapping of space.
    ```python
    def torus(self):
        # Retrieve current position and adjust according to the torus world rules
        robot_position_x, robot_position_y, robot_heading = self.agent.get_position()
        robot_position_x = robot_position_x % self.config['world_width']
        robot_position_y = robot_position_y % self.config['world_height']
        self.agent.set_position(robot_position_x, robot_position_y, robot_heading)
    ```

  - **Subtask 1.2**: Integrated light sensors in `my_world`.
    - Established a method to define light distribution in the environment:
    ```python
    def defineLight(self):
        light_source_position = np.array([self.config['world_width'] / 2, self.config['world_height'] / 2])
        max_light_intensity = 60
        light_dist = np.zeros((self.config['world_width'], self.config['world_height']))
        for x in range(self.config['world_width']):
            for y in range(self.config['world_height']):
                distance = np.linalg.norm(np.array([x, y]) - light_source_position)
                light_dist[x, y] = max(0, max_light_intensity - distance/3)
        return light_dist
    ```

#### Task 1.2: Implementation of Braitenberg Vehicles
  - **Subtask 1.1**: Created light sensors for each side (left and right).
    - Developed modules `light_sensor_L.py` and `light_sensor_R.py` containing the class implementations `LightIntensitySensor_L` and `LightIntensitySensor_R`.
  - **Subtask 1.2**: Developed two types of controllers: Aggressor and Fear.
    - For the Aggressor vehicle, the robot's behavior is guided by differential light intensity:
    ```python
    sensor = self.agent.get_perception()
    delta_s = abs(sensor[1] - sensor[2])
    base_speed = 5
    sensitivity = 50
    speed = base_speed + sensitivity * delta_s
    if sensor[1] > sensor[2]:
        turn_angle = 5
    elif sensor[1] < sensor[2]:
        turn_angle = -5
    else:
        turn_angle = 0
    ```
    - The Fear vehicle utilizes the opposite strategy for turn angles.

### Incomplete Tasks
- Once Agressor Vehicle got close to light source, it keeps wandering around light source. This prevent Agressor Vehicle from getting out of light source and driving the torus around freely.


## Additional Documentation
- **Plots**: All plots are stored in the `plots/` directory in PNG format. These plots illustrate light distribution and the trajectories of both types of vehicles.
- **Videos**: Demonstrative videos showcasing the behavior of each vehicle type are available in the `videos/` directory.

## Additional Notes

---
