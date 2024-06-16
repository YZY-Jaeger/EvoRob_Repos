import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the column names and read the data
column_names = ["num_example", "class", "feature x", "feature y"]
df = pd.read_csv('data', sep=' ', names=column_names)

# Activation function
def activation_function(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

# Simple Artificial Neural Network class
class SimpleANN:
    def __init__(self, weights=None, bias=None):
        if weights is None:
            self.weights = np.random.rand(2)  #Initialize weights randomly
        else:
            self.weights = weights
        if bias is None:
            self.bias = np.random.rand(1)  #Initialize bias randomly
        else:
            self.bias = bias

    def predict(self, df):
        inputs = df[["feature x", "feature y"]].values
        total = np.dot(inputs, self.weights) - self.bias#w1x1 + w2x2 − w0 = output
        output = activation_function(total)
        return np.where(output < 0, 0, 1) #where(condition, [x, y])
                                        #‘class 0’ if it outputs φ < 0 and as ‘class 1’ if it outputs φ > 0.

    def fitness(self, df):
        actual = df["class"].values
        predicted = self.predict(df)
        return np.mean(actual == predicted)

# Genetic Algorithm for evolving ANNs
def evolve_anns(df, generations=100, population_size=50, mutation_rate=0.1, elitism_size=5):
    population = [SimpleANN() for _ in range(population_size)]
    fitness_history = []

    for generation in range(generations):
        fitnesses = np.array([ann.fitness(df) for ann in population])
        fitness_history.append(fitnesses)
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        
        # Elitism: keep the best 5 parents
        new_population = [ranked[i][0] for i in range(elitism_size)]

        # Select individuals based on rank for crossover
        selection_prob = np.linspace(1, 0, population_size)
        selection_prob = selection_prob / selection_prob.sum()
        selected_indices = np.random.choice(population_size, population_size - elitism_size, p=selection_prob)
        selected = [ranked[i][0] for i in selected_indices]

        # Crossover and mutation to fill up the rest of the population
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(selected, 2, replace=False)
            child_weights = (parent1.weights + parent2.weights) / 2
            child_bias = (parent1.bias + parent2.bias) / 2
            # Mutation for weights
            if np.random.rand() < mutation_rate:
                mutation_index = np.random.randint(2)
                child_weights[mutation_index] += np.random.normal(0, 0.1)
            # Mutation for bias
            if np.random.rand() < mutation_rate:
                child_bias += np.random.normal(0, 0.1)
            new_population.append(SimpleANN(weights=child_weights, bias=child_bias))

        population = new_population

    best_ann = sorted(zip(population, [ann.fitness(df) for ann in population]), key=lambda x: x[1], reverse=True)[0][0]
    return best_ann, fitness_history

# Evolve the ANNs
best_ann, fitness_history = evolve_anns(df)

# Print the best weights and bias
# Print the best weights and bias
w1, w2 = best_ann.weights
w0 = best_ann.bias[0]  
print("Best weights: w1 = {}, w2 = {}".format(w1, w2))
print("Best bias: w0 = {}".format(w0))


# Plot fitness history
plt.plot(np.max(fitness_history, axis=1), label='Max Fitness')
plt.plot(np.mean(fitness_history, axis=1), label='Average Fitness')
plt.title('Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Max Fitness')


# Save plot
plot_filename = "plot1_Fitness over Generations.png"
plt.savefig(plot_filename)
print(f"Plot saved as {plot_filename}")

plt.show()

# Function to plot decision boundary
def plot_decision_boundary(ann, df):
    x_min, x_max = df["feature x"].min() - 1, df["feature x"].max() + 1
    y_min, y_max = df["feature y"].min() - 1, df["feature y"].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    df_grid = pd.DataFrame(grid, columns=["feature x", "feature y"])
    Z = ann.predict(df_grid).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(df["feature x"], df["feature y"], c=df["class"], edgecolors='k')
    plt.title('Decision Boundary')

    # Save plot
    plot_filename = "plot2_Decision Boundary.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    plt.show()

# Plot the decision boundary
plot_decision_boundary(best_ann, df)
