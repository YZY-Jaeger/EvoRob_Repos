# EvoRobotics Group Assignment

## Team Members
- **Minsol Kim**
- **Yu Zeyuan**

## Project Files and Directory Structure

- `EvoRob03.py`: Contains the main genetic algorithm.
- `plots/`: Directory containing all the generated plots.
  - `fitness_distribution.png`: Plot of the distribution of generations needed for simulations being run.
  - 'Fitness Progression over Generations.png': Plot of the fitness progression over generations.

## Tasks Completed

- **Algorithm Implementation**: Developed the Python script to perform genetic string evolution using a genetic algorithm.
- **Fitness Evaluation**: Implemented fitness evaluation using the `SequenceMatcher` from Python's `difflib` module.
- **Plot Generation**: Generated plots showing the distribution of generations needed to achieve the target string and the fitness progress over generations of the first simulation.
- **Mathematical Estimation**: Provided a mathematical estimation of the expected number of generations required to match the target string.
- **Empirical Simulation**: Conducted multiple simulations to empirically determine the average number of generations needed, comparing it to the mathematical estimation.

## Answer to "Good-Natured" Optimization Problem
This problem is described as "good-natured" because it involves optimizing a playful scenario using a model that mimics natural evolutionary processes. The "fitness landscape" in this context is relatively straightforward and benign, focused solely on increasing alignment with a fixed target string through slight, random changes. This contrasts with more competitive or survival-based applications of genetic algorithms in complex, multi-variable optimization landscapes.

## Mathematical Estimation and Empirical Comparison
**Estimation**:
Given a target string of 30 characters and a character set of 27 (26 letters + space), the probability of any single character mutating correctly is 1/27. Assuming independence, the expected number of mutations for one correct character can be approximated as 27 mutations. For the string to entirely match, it would naively be expected to take about 27*30 = 810 mutations in a purely random process *without selection mechanisms*.

This simple calculation doesn't take selection mechanism into our calculation, since the successful mutation of one letter could effect to other letter's selection and mutation.

#### Empirical Data
Our empirical finding that it takes around 23,000 generations on average suggests that the mutation and selection process is less efficient than our model might hope. This could be due to:
- High dependence on initial random strings.
- A low mutation rate or low retention rate of beneficial mutations.
- One mutation affects to mutation for other mutation following.


From our simulations, the average number of generations needed to reach the goal string across 10 trials was approximately 23000. It sounds like there's a need to revisit and refine the mathematical estimation of the expected number of generations needed for the genetic algorithm to evolve the target string. We assumed this is because of above three reasons, especially the last one. 
As an evidence, if we see the plot named 'Fitness Progression over Generations for the first simulation', we can observe the progression is not steady but all at sudden. This proves that each mutation is not independent to those of previous ones.

### Conclusion
Revisiting our simulation setup or considering different strategies in mutation and selection might help in understanding the disparity in the expected versus observed number of generations. Adjusting parameters like mutation rates or implementing strategies like crossover might also alter the dynamics and efficiency of our evolutionary algorithm.
