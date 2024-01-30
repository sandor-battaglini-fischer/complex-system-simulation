import numpy as np
import random
from scipy.integrate import solve_ivp
def random_starting_matrix():
    A = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            A[i, j] = round(random.uniform(-1, 1), 2)
    return A

def random_starting_vector():
    A = np.ones((4))
    for i in range(4):
        A[i] = round(random.uniform(-1, 1), 2)
    return A

def xy_starting_vector():
    A = np.zeros((4))
    return A