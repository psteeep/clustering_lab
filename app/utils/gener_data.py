import numpy as np

def generate_random_points(n_points, min_val=0, max_val=100):
    return np.random.uniform(min_val, max_val, size=(n_points, 2))