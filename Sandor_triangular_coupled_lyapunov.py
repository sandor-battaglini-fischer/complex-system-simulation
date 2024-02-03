import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import time
from multiprocessing import Pool

""" 
Model of Love Dynamics 
from chapter 14 and 15 of the book.

Model of Kathe-Jules-Jim triangular love dynamics.

"""

central_partner = "Kathe"
lover1 = "Jules"
lover2 = "Jim"


def largest_lyapunov_exponent(initial_conditions, epsilon, delta, params, T=208, dt=0.02):
    t = np.arange(0, T, dt)
    n = len(t)
    
    perturbed_initial = initial_conditions + np.random.normal(0, delta, len(initial_conditions))

    sol1 = odeint(love_dynamics, initial_conditions, t, args=(params,))
    sol2 = odeint(love_dynamics, perturbed_initial, t, args=(params,))

    divergence = np.linalg.norm(sol2 - sol1, axis=1)
    small_constant = 1e-15
    divergence = np.maximum(divergence, small_constant)
    lyapunov = 1/n * np.sum(np.log(divergence/delta))

    return lyapunov


def compute_LLE_for_params(param_tuple):
    epsilon, delta = param_tuple
    print(f"Processing epsilon: {epsilon:.4f}, delta: {delta:.4f}")
    return largest_lyapunov_exponent(initial_conditions, epsilon, delta, params)



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
    alpha1, alpha2, alpha3, beta21, beta12, beta13, beta31, gamma1, gamma2, gamma3, epsilon, delta, A1, A2, A3, tauI12, sigmaL12, sigmaI12, beta12, tau_S, sigmaS, tauP, p, sigmaP, tauI31, sigmaL31, sigmaI31, s = params

    dx12dt = -alpha1 * np.exp(epsilon * (x13 - x12)) * x12 + RL12(x21, tauI12, sigmaL12, sigmaI12, beta12) + (1 + S(x12, tau_S, sigmaS, s)) * gamma1 * A2
    dx13dt = -alpha1 * np.exp(epsilon * (x12 - x13)) * x13 + beta13 * x31 + (1 + S(x13, tau_S, sigmaS, s)) * gamma1 * A3
    dx21dt = -alpha2 * x21 + beta21 * x12 * np.exp(delta * (x13 - x12)) + (1 - P(x21, tauP, p, sigmaP)) * gamma2 * A1
    dx31dt = -alpha3 * x31 + RL31(x13, tauI31, beta31, sigmaL31, sigmaI31) * np.exp(delta * (x13 - x12)) + gamma3 * A1

    return [dx12dt, dx13dt, dx21dt, dx31dt]

# Parameters w/ extensive comments
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
    8,    # beta12: reaction coefficient to love for Kathe to Jules' love
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
t = np.linspace(0, 20, 100) 
solution = odeint(love_dynamics, initial_conditions, t, args=(params,))


if __name__ == '__main__':
    epsilon_values = np.linspace(0, 0.015, 100)
    delta_values = np.linspace(0, 0.045, 100)

    param_combinations = [(epsilon, delta) for epsilon in epsilon_values for delta in delta_values]

    start_time = time.time()

    with Pool() as pool:
        results = pool.map(compute_LLE_for_params, param_combinations)

    LLE_values = np.array(results).reshape(len(epsilon_values), len(delta_values))

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(LLE_values, cmap='seismic', vmin=-1, vmax=1, extent=[epsilon_values.min(), epsilon_values.max(), delta_values.min(), delta_values.max()], aspect='auto', origin='lower')
    plt.colorbar(label='Largest Lyapunov Exponent')
    plt.xlabel('Epsilon')
    plt.ylabel('Delta')
    plt.title(r'Heatmap of LLE for different Epsilon and Delta')
    plt.show()

