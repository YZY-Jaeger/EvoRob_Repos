import numpy as np
import matplotlib.pyplot as plt
from evolution2 import evolutionary_algorithm

# Parameters to test
pop_sizes = [30, 50, 100]
mutation_rates = [0.005, 0.05, 0.1]
num_generations = 100  # Fixed
tournament_sizes = [2, 3, 5]
prob_crossover = 0.9  # Fixed
elite_size = 5        # Fixed

# Iterate through all population sizes
for pop_size in pop_sizes:
    # Create a new plot window
    plt.figure(figsize=(15, 7))
    
    # Iterate through all mutation rates
    for i, mutation_rate in enumerate(mutation_rates, start=1):
        plt.subplot(1, 3, i)  # Subplot for each mutation rate
        for tournament_size in tournament_sizes:
            # Run the evolutionary algorithm
            best_fitnesses, avg_fitnesses = evolutionary_algorithm(
                pop_size, num_generations, mutation_rate, prob_crossover, tournament_size, elite_size)

            # Plot results for each tournament size
            plt.plot(best_fitnesses, label=f'Best Fitness (tsize={tournament_size})')
            plt.plot(avg_fitnesses, label=f'Avg Fitness (tsize={tournament_size})')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'pop_size={pop_size}, mrate={mutation_rate}')
        plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plot_filename = f"plot_psize{pop_size}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    # Show plot
    plt.show()
