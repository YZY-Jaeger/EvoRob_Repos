# =============================================================================
# created by:   Samer Al-Magazachi
# created on:   06/04/2021 -- 13/04/2022
# version:      0.9
# status:       prototype
# =============================================================================
import yaml
from swarmy.experiment import Experiment

### load the configuration file, check the config.yaml file for more information and to change to your needs
with open ('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
## Import uour implementation of the controller, sensor, environment and agent
from controller.ANN_controller import EvolutionaryANNController
from sensors.proximity_sensors import ProximitySensor
from world.example_world2 import My_environment
from agent.my_agent import MyAgent
from evolution.evolution import *
import random
import matplotlib.pyplot as plt


# add your controller, if you have more than one controller, add them to the list and specify the percentage of robots that should use this controller in the config.yaml file
agent_controller = [EvolutionaryANNController]
# add your sensors, if you have more than one sensor, add them to the list all sensors are added to each robot
agent_sensing = [ProximitySensor]


""" initialize your evolution object and start the evolution process by initializing the first genome 
    and then run the experiment for each generation, evaluate the fitness of the genome and update the genome.

    One generation could look like this:
    - initialize the experiment with: exp = Experiment(config, agent_controller, agent_sensing, My_environment, MyAgent)
    - initialize the robots with: exp.init_robots()
    - set the control parameters of the robot with: exp.agent_list[0].set_control_params(genome) 
        here, the genome could be a list of floating point values that are used in the controller class
    - run the experiment with: exp.run(config['rendering'])
    - use ecp.agent_list[0].get_evaluation_params() to get the information you need to calculate the the fitness 
    (don't forget to overwrite the get_evaluation_params() function in your agent class to your needs)
    - calculate the fitness and mutate
    - repeat for each generation
"""

generation_size = 10
ann_num = 12
elitism_size = 4
Anns = [ANN(input_size=3, hidden_size=2, output_size=2) for _ in range(ann_num)]
fitnesses = [0 for _ in range(ann_num)]
mutation_rate = 0.3
best_fitnesses = []
avg_fitnesses = []

deterministic = 1

for gen in range(generation_size):
    
    if deterministic:
        for a in range(ann_num):
            exp = Experiment(config, agent_controller, agent_sensing, My_environment, MyAgent)
            exp.init_robots()
            exp.agent_list[0].set_control_params(Anns[a])
            exp.agent_list[0].set_position(330, 330, 0)
            print("----------------------------------------------------")

            exp.run(config['rendering'])
            fitnesses[a] = Evolution.calculate_fitness(exp.agent_list[0]) - exp.agent_list[0].mistake * 0.001
            print(f"Generation {gen} ANN {a} Fitness: {fitnesses[a]}")
    else:
        for a in range(ann_num):
            exp = Experiment(config, agent_controller, agent_sensing, My_environment, MyAgent)
            exp.init_robots()
            exp.agent_list[0].set_control_params(Anns[a])
            print("----------------------------------------------------")

            exp.run(config['rendering'])
            fitnesses[a] = Evolution.calculate_fitness(exp.agent_list[0]) - exp.agent_list[0].mistake * 0.0001
            print(f"Generation {gen} ANN {a} Fitness: {fitnesses[a]}")
    
    '''
    for f in range(len(fitnesses)):
        if fitnesses[f] <=0:
            fitnesses[f] = 0# to avoid negative fitness
    '''
    # Calculate and print the best and average fitness for the current generation
    best_fitness = max(fitnesses)
    avg_fitness = sum(fitnesses) / len(fitnesses)
    print(f"Best Fitness in Generation {gen}: {best_fitness}")
    print(f"Average Fitness in Generation {gen}: {avg_fitness}")
    # Store the fitness values
    best_fitnesses.append(best_fitness)
    avg_fitnesses.append(avg_fitness)


    sorted_fitnesses = sorted(fitnesses, reverse=True)   
    '''
    for f in range(len(sorted_fitnesses)):
        if sorted_fitnesses[f] <=0:
            sorted_fitnesses[f] = 0# to avoid negative fitness
    '''
    sum_fitnesses = sum(sorted_fitnesses)
    sorted_fitnesses = [fit / sum_fitnesses for fit in sorted_fitnesses]#normalize fitnesses

    sorted_pop = sorted(zip(Anns, fitnesses), key=lambda x: x[1], reverse=True)

    # Rank-based selection preparation
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    ranks = [sorted_indices.index(i) + 1 for i in range(len(fitnesses))]
    total_ranks = sum(ranks)
    selection_probabilities = [(total_ranks - rank + 1) / total_ranks for rank in ranks]
    selection_probabilities_sum = sum(selection_probabilities)
    selection_probabilities = [p / selection_probabilities_sum for p in selection_probabilities]
    print(selection_probabilities)
    elites = [Anns[i] for i in sorted_indices[:elitism_size]]
    #elites = [ind for ind, fit in sorted_pop[:elitism_size]]

    # Extract just the ANNs for selection
    sorted_anns = [ann for ann, _ in sorted_pop]

        # Generate new population with mutation and elitism
    new_population = elites[:]
    while len(new_population) < ann_num:
        parent1, parent2 = np.random.choice(sorted_anns, 2, replace=False, p = selection_probabilities)
        offspring = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            mutate(offspring)
        new_population.append(offspring)

    Anns = new_population
    exp.agent_list[0].current_generation += 1

plt.figure(figsize=(10, 5))
plt.plot(best_fitnesses, label='Best Fitness')
plt.plot(avg_fitnesses, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Progression Over Generations')
plt.legend()
plt.show()