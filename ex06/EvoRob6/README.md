# EvoRobotics Group Assignment

## Team Members
- **Minsol Kim**
- **Yu Zeyuan**

## Project Files and Directory Structure
- EvoRob_EX06.py
- plot1_Fitness over Generations.png
- plot2_Decision Boundary.png

## Implementation Details
### Implement a complete evolutionary algorithm (Elitism)

Last time, we unintentionally used algorithm that resembles 'hill climber',putting and comparing some parents together, not killing some of them. However this time, we used correct method of elitism replacement.

- Population Initialization: A population of ANNs is initialized with random weights and biases.
- Fitness Evaluation: The fitness of each ANN is evaluated based on its classification accuracy on the dataset.
- Selection: The top-performing ANNs are selected based on their fitness (elitism).
- Crossover and Mutation: Selected ANNs are combined to produce offspring, and mutations are introduced to maintain diversity.
- Elitism: The best-performing ANNs are preserved across generations.

### a simple ANN (without a hidden layer)

- The ANN has two input neurons corresponding to the features x and y, one output neuron, and a bias neuron.
- The fitness of the ANN is determined by the proportion of correctly classified examples in the dataset.

- The best weights and bias found:
(vary each time the script is executed; they will be printed in the terminal):
    - Weights: [w1, w2]
    - Bias: w0


