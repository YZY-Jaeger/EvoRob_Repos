import numpy as np

def ackley_function(x, y, z):
    part1 = -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2 + z**2) / 3))
    part2 = -np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)) / 3)
    return part1 + part2 + 20 + np.exp(1)

def fitness(x, y, z):
    return 1 / (ackley_function(x, y, z) + 1)







