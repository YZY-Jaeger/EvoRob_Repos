# EvoRob 07

## Team Members
- **Minsol Kim**
- **Yu Zeyuan**

## Project Files and Directory Structure
- EvoRob_EX07.py
- XOR_NN_Pop1000_Gen2000.png

## Implementation Details
### Neural Network

In this assignment, the neural network has been upgraded to include one hidden layer with 2 neurons. The architecture is as follows:

Input Layer: 2 neurons
Hidden Layer: 2 neurons
Output Layer: 1 neuron
Compared to last task sheet, now ANN has one hidden layer with 2 neurons. 

###Key Improvements
- **Optimization of Fitness Evaluation**:

Initially, the program was running slowly due to repeated calls to the evaluate_fitness function within the rank-based selection process.
By calculating the fitness once per generation and storing it in a variable, we significantly improved the performance.

- **Weight Initialization**:

The initial fitness scores were capped at around 0.75, which was suboptimal.
This issue was traced back to the weight initialization range, which was set between 0 and 1.
By adjusting the weight initialization range to be between -1 and 1, the neural network's performance improved, achieving fitness levels above 0.99.


### Genetic Algorithm
The genetic algorithm evolves the neural network to solve the XOR problem. Key steps include:

- Initialization: Create a population of neural networks with randomly initialized weights.
- Fitness Evaluation: Measure each network's performance on the XOR problem.
- Rank-Based Selection: Select parents for the next generation based on their fitness ranks.
- Crossover: Combine parent weights to create child networks.
- Mutation: Apply random changes to the weights of child networks.
- Elitism: Preserve the top-performing networks for the next generation.

