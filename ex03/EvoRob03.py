import os
import random
import string
from difflib import SequenceMatcher
import matplotlib.pyplot as plt


goal = "charles darwin was always seasick"
length_of_goal = len(goal)
simulations = 10 # Number of simulations to run
generations_needed = []
fitness_history = []  # List to store fitness values

def get_random_string(length):
    letters = string.ascii_lowercase + " "
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

for simulation in range(simulations):
    generation = 0
    print(f"========== Simulation {simulation} started ==========")
    default_str = get_random_string(length_of_goal)
    similarity = int(SequenceMatcher(None, default_str, goal).ratio() * 100)
    
    if simulation == 0:
        fitness_history.append(similarity)  # Add initial fitness to history

    while similarity < 100:
        temp_str = default_str
        new_letter = random.choice(string.ascii_lowercase + " ")
        random_index = random.randint(0, len(default_str) - 1)
        temp_str = default_str[:random_index] + new_letter + default_str[random_index + 1:]
        new_sim = int(SequenceMatcher(None, temp_str, goal).ratio() * 100)

        if new_sim >= similarity:
            default_str = temp_str
            similarity = new_sim
            generation += 1
            if simulation == 0:
                fitness_history.append(similarity)  # Store fitness for each generation
            print("#"+str(generation)+" "+str(similarity) + "%"+" "+default_str)

            
    generations_needed.append(generation)

average_generations = sum(generations_needed) / len(generations_needed)
print(f"Average generations needed: {average_generations:.2f}")

# Plotting the fitness history for the first simulation
plt.figure(figsize=(10, 5))
plt.plot(fitness_history, marker='o', linestyle='-', color='b')
plt.title('Fitness Progression over Generations for the first simulation')
plt.xlabel('Generation')
plt.ylabel('Fitness (%)')
plt.grid(True)

# Save the plot
plot_path = os.path.join('plots', 'Fitness Progression over Generations.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

plt.show()



# Plotting results
plt.figure(figsize=(10, 5))
plt.hist(generations_needed, bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Generations Needed')
plt.xlabel('Generations Needed')
plt.ylabel('Frequency')
plt.grid(True)

text_x = 0.95  
text_y1 = 0.85  
text_y2 = 0.80
plt.text(text_x, text_y1, f'Total simulations: {simulations}', fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)
plt.text(text_x, text_y2, f'Average generations needed: {average_generations:.2f}', fontsize=12, ha='right', va='top', transform=plt.gca().transAxes, color='black')


# Save the plot
plot_path = os.path.join('plots', 'fitness_distribution.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

plt.show()

