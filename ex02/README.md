# EvoRobotics Group Assignment

## Team Members
- **Minsol Kim**
- **Yu Zeyuan**

## Overview
This repository hosts our group's submission for Task Sheet 2 of the Evolutionary Robotics course. The submission includes all necessary code, trajectory image, and supplementary resources essential for evaluating our implementation of specified robotic behaviors and sensor integrations.

## Tasks and Subtasks
### Completed Tasks

#### Task 2.1:  Proximity Sensor

  - **/controller/proximity_sensor.py**: implemented sensor, returning distance detection from 3 direction
    ```python
    class ProximitySensor(Perception):
    ```
  

  - **Environment set up**: randomly generated walls, in my_world.py
    ```python
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
    ```

#### Task 1.2:  Control by Rules
  - **in Controller_proximity.py**: if conditions controlling robot to go and explore
```python
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
        speed = 1 #go

    self.update_position(speed, turn_angle)
```


### Incomplete Tasks
- 


## Additional Documentation
- **Plots**: trajectory.png recording the line that robot passed
- **Videos**: Demonstrative videos in the same directory with workspace.py

## Additional Notes

- Walls are randomly generated so sometimes the robot get stuck, just re-start program to avoid this
