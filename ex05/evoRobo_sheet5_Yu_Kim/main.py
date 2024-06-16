import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from evolution import evolutionary_algorithm


pop_size = 50
bounds = [-32.768, 32.768]
mutation_rate = 0.1
num_generations = 100
elite_size = 2

best_fitnesses, avg_fitnesses = evolutionary_algorithm(pop_size, num_generations)

plt.plot(best_fitnesses, label='Best Fitness')
plt.plot(avg_fitnesses, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title(f'pop_size={pop_size}, mutation_rate={mutation_rate}, num_generations={num_generations}')
plt.legend()


# Save plot
plot_filename = "plot1.png"
plt.savefig(plot_filename)
print(f"Plot saved as {plot_filename}")

# Show plot
plt.show()

'''
def ackley_function_2d(x, y):
    part1 = -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2) / 2))
    part2 = -np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2)
    return part1 + part2 + 20 + np.exp(1)

x = np.linspace(-32.768, 32.768, 400)
y = np.linspace(-32.768, 32.768, 400)
x, y = np.meshgrid(x, y)

z = ackley_function_2d(x, y)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Ackley Function Value')
ax.set_title('Ackley Function in 2D')

plt.show()
'''