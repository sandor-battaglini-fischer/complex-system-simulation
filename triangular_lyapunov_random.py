import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import time
from multiprocessing import Pool
from scipy.optimize import curve_fit
from numpy import random
import pandas as pd

""" 
Model of Love Dynamics 
from chapter 14 and 15 of the book.

Model of Kathe-Jules-Jim triangular love dynamics.

"""

central_partner = "Kathe"
lover1 = "Jules"
lover2 = "Jim"

def largest_lyapunov_exponent(initial_conditions, var, params, T=208, dt=0.02):
    t = np.arange(0, T, dt)
    n = len(t)

    # Generate random perturbations for each state variable
    perturbed_initial = initial_conditions + np.random.normal(0, var, len(initial_conditions))

    sol1 = odeint(love_dynamics, initial_conditions, t, args=(params,))
    sol2 = odeint(love_dynamics, perturbed_initial, t, args=(params,))

    divergence = np.linalg.norm(sol2 - sol1, axis=1)
    small_constant = 1e-15
    divergence = np.maximum(divergence, small_constant)
    lyapunov = 1/n * np.sum(np.log(divergence/var))

    return lyapunov


def compute_LLE_for_random_params(initial_conditions, var, params):
    return largest_lyapunov_exponent(initial_conditions, var, params)



# Kathe's reaction function to love from Jules
def RL12(x21, tauI12, sigmaL12, sigmaI12, beta12):
    if x21 >= tauI12:
        return beta12 * x21 / (1 + x21/sigmaL12) * (1 - ((x21 - tauI12) / sigmaI12)**2) / (1 + ((x21 - tauI12) / sigmaI12)**2)
    else:
        return beta12 * x21 / (1 + x21/sigmaL12)

# Synergism function of Kathe (j=2,3)
def S(x1j, tau_S, sigmaS, s):
    if x1j >= tau_S:
        return s*((x1j - tau_S) / sigmaS)**2 / (1 + ((x1j - tau_S) / sigmaS)**2)
    else:
        return 0

# Platonicity function as defined by Jules
def P(x21, tauP, p, sigmaP):
    if x21 >= tauP:
        return p*((x21 - tauP) / sigmaP)**2 / (1 + ((x21 - tauP) / sigmaP)**2)
    else:
        return 0

# Jim's reaction function to love from Kathe
def RL31(x13, tauI31, beta31, sigmaL31, sigmaI31):
    if x13 >= tauI31:
        return beta31 * x13 / (1 + x13/sigmaL31) * (1 - ((x13 - tauI31) / sigmaI31)**2) / (1 + ((x13 - tauI31) / sigmaI31)**2)
    else:
        return beta31 * x13 / (1 + x13/sigmaL31)


def love_dynamics(y, t, params):
    x12, x13, x21, x31 = y
    alpha1, alpha2, alpha3, beta21, beta12, beta13, beta31, gamma1, gamma2, gamma3, epsilon, delta, A1, A2, A3, tauI12, sigmaL12, sigmaI12, tau_S, sigmaS, tauP, p, sigmaP, tauI31, sigmaL31, sigmaI31, s = params

    dx12dt = -alpha1 * np.exp(epsilon * (x13 - x12)) * x12 + RL12(x21, tauI12, sigmaL12, sigmaI12, beta12) + (1 + S(x12, tau_S, sigmaS, s)) * gamma1 * A2
    dx13dt = -alpha1 * np.exp(epsilon * (x12 - x13)) * x13 + beta13 * x31 + (1 + S(x13, tau_S, sigmaS, s)) * gamma1 * A3
    dx21dt = -alpha2 * x21 + beta21 * x12 * np.exp(delta * (x13 - x12)) + (1 - P(x21, tauP, p, sigmaP)) * gamma2 * A1
    dx31dt = -alpha3 * x31 + RL31(x13, tauI31, beta31, sigmaL31, sigmaI31) * np.exp(delta * (x13 - x12)) + gamma3 * A1

    return [dx12dt, dx13dt, dx21dt, dx31dt]

params = [
    2,    # alpha1: forgetting coefficient for Kathe (years^-1)
    1,    # alpha2: forgetting coefficient for Jules (years^-1)
    2,    # alpha3: forgetting coefficient for Jim (years^-1)
    1,    # beta21: reaction coefficient to love for Jules to Kathe's love(years^-1)
    8,    # beta12: reaction coefficient to love for Kathe to Jules love (years^-1)
    1,    # beta13: reaction coefficient to love for Kathe to Jim's love (years^-1)
    2,    # beta31: reaction coefficient to love for Jim to Kathe's love (years^-1)
    1,    # gamma1: reaction coefficient to appeal for Kathe (years^-1)
    0.5,  # gamma2: reaction coefficient to appeal for Jules (years^-1)
    1,    # gamma3: reaction coefficient to appeal for Jim (years^-1)
    0.0062,   # epsilon: sensitivity of reaction to love for Kathe (coupling constant)
    0.0285,    # delta: sensitivity of reaction to love for Jules and Jim (coupling constant)
    20,   # A1: appeal of Kathe (dimensionless)
    4,    # A2: appeal of Jules (dimensionless)
    5,    # A3: appeal of Jim (dimensionless)
    2.5,  # tauI12: insecurity threshold for Kathe's reaction to Jules' love
    10,   # sigmaL12: sensitivity of reaction to love for Kathe to Jules
    10.5, # sigmaI12: sensitivity of insecurity for Kathe to Jules
    9,    # tau_S: synergism threshold for Kathe
    1,    # sigmaS: sensitivity of synergism for Kathe
    0,    # tauP: platonicity threshold for Jules
    1,    # p: maximum platonicity for Jules
    1,    # sigmaP: sensitivity of platonicity for Jules
    9,    # tauI31: insecurity threshold for Jim's reaction to love
    10,   # sigmaL31: sensitivity of reaction to love for Jim
    1,    # sigmaI31: sensitivity of insecurity for Jim
    2,    # s: synergism coefficient for Kathe
]



initial_conditions = [0, 0, 0, 0]
t = np.linspace(0, 20, 1000) 
solution = odeint(love_dynamics, initial_conditions, t, args=(params,))

def random_parameters_sampling(num_samples):
    alpha1_range = (1, 5)
    alpha2_range = (1, 5)
    alpha3_range = (1, 5)
    beta21_range = (1, 10)
    beta12_range = (1, 10)
    beta13_range = (1, 10)
    beta31_range = (1, 10)
    gamma1_range = (0.5, 1.5)
    gamma2_range = (0.5, 1.5)
    gamma3_range = (0.5, 1.5)
    epsilon_range = (0.001, 0.01)
    delta_range = (0.01, 0.05)
    A1_range = (1, 20)
    A2_range = (1, 20)
    A3_range = (1, 20)
    tauI12_range = (1, 10)
    sigmaL12_range = (1, 10)
    sigmaI12_range = (1, 10)
    tau_S_range = (1, 10)
    sigmaS_range = (1, 10)
    tauP_range = (1, 10)
    p_range = (1, 10)
    sigmaP_range = (1, 10)
    tauI31_range = (1, 10)
    sigmaL31_range = (1, 10)
    sigmaI31_range = (1, 10)
    s_range = (1, 10)
    

    # Random sampling
    sampled_params = []
    for _ in range(num_samples):
        alpha1 = random.uniform(*alpha1_range)
        alpha2 = random.uniform(*alpha2_range)
        alpha3 = random.uniform(*alpha3_range)
        beta21 = random.uniform(*beta21_range)
        beta12 = random.uniform(*beta12_range)
        beta13 = random.uniform(*beta13_range)
        beta31 = random.uniform(*beta31_range)
        gamma1 = random.uniform(*gamma1_range)
        gamma2 = random.uniform(*gamma2_range)
        gamma3 = random.uniform(*gamma3_range)
        epsilon = random.uniform(*epsilon_range)
        delta = random.uniform(*delta_range)
        A1 = random.uniform(*A1_range)
        A2 = random.uniform(*A2_range)
        A3 = random.uniform(*A3_range)
        tauI12 = random.uniform(*tauI12_range)
        sigmaL12 = random.uniform(*sigmaL12_range)
        sigmaI12 = random.uniform(*sigmaI12_range)
        tau_S = random.uniform(*tau_S_range)
        sigmaS = random.uniform(*sigmaS_range)
        tauP = random.uniform(*tauP_range)
        p = random.uniform(*p_range)
        sigmaP = random.uniform(*sigmaP_range)
        tauI31 = random.uniform(*tauI31_range)
        sigmaL31 = random.uniform(*sigmaL31_range)
        sigmaI31 = random.uniform(*sigmaI31_range)
        s = random.uniform(*s_range)
  
        
        # Append the sampled parameter set
        sampled_params.append([alpha1, alpha2, alpha3, beta21, beta12, beta13, beta31, gamma1, gamma2, gamma3, epsilon, delta, A1, A2, A3, tauI12, sigmaL12, sigmaI12, tau_S, sigmaS, tauP, p, sigmaP, tauI31, sigmaL31, sigmaI31, s])

    return sampled_params


if __name__ == '__main__':
    num_samples = 100000
    initial_conditions = [0, 0, 0, 0]
    sampled_params = random_parameters_sampling(num_samples)
    var = 1e-6

    start_time = time.time()

    with Pool() as pool:
        results = pool.starmap(compute_LLE_for_random_params, [(initial_conditions, var, params) for params in sampled_params])

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

    # Filter for positive LLEs and keep corresponding parameters and LLEs
    positive_LLE_data = [(params, lle) for params, lle in zip(sampled_params, results) if lle > 0 and lle < 1e8]

    if not positive_LLE_data:
        print("No positive LLEs found.")
    else:
        positive_params, positive_LLEs = zip(*positive_LLE_data)

        param_names = ["alpha1", "alpha2", "alpha3", "beta21", "beta12", "beta13", "beta31", "gamma1", "gamma2", "gamma3", "epsilon", "delta", "A1", "A2", "A3", "tauI12", "sigmaL12", "sigmaI12", "tau_S", "sigmaS", "tauP", "p", "sigmaP", "tauI31", "sigmaL31", "sigmaI31", "s"]
        df_positive = pd.DataFrame(positive_params, columns=param_names)
        df_positive['LLE'] = positive_LLEs  

        param_ranges = df_positive.agg(['min', 'max'])
        print("Parameter value ranges for positive LLEs:")
        print(param_ranges)

        # Analyze sensitivity of parameters
        param_std = df_positive[param_names].std()
        sensitive_params = param_std.sort_values(ascending=False)
        print("Parameters sensitivity ranking for producing positive LLEs:")
        print(sensitive_params)

        # Save to Excel
        df_positive.to_excel("positive_love_dynamics_LLEs_all.xlsx", index=False)
        param_ranges.to_excel("positive_love_dynamics_LLEs_param_ranges.xlsx", index=True)
        sensitive_params.to_excel("positive_love_dynamics_LLEs_param_sensitivity.xlsx", index=True)
