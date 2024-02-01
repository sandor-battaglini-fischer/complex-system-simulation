import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

def love_dynamics(y, t, params):
    alpha1, alpha2, beta1, beta2, gamma1, gamma2, A1, A2, x_star2, epsilon, mu, delta = params
    x1, x2, z2 = y

    RL1 = beta1 * x2 * (1 - (x2 / x_star2)**2)
    RL2 = beta2 * x1 + gamma2 * A1 / (1 + delta * z2)
    dz2dt = epsilon * (mu * x2 - z2)

    dx1dt = -alpha1 * x1 + RL1 + gamma1 * A2
    dx2dt = -alpha2 * x2 + RL2

    return [dx1dt, dx2dt, dz2dt]

def largest_lyapunov_exponent(initial_conditions, a1, delta, params, omega, perturbation_delta=0.0001, T=208, dt=0.02):
    t = np.arange(0, T, dt)
    n = len(t)
    
    # Adjust the parameter for A1 within the params list before passing it to odeint
    updated_params = list(params)
    updated_params[6] = a1  # Assuming that A1 is at index 6 in the params list

    # Create the perturbed initial conditions
    perturbed_initial = initial_conditions + np.random.normal(0, perturbation_delta, len(initial_conditions))

    # Call odeint with the updated_params
    sol1 = odeint(love_dynamics, initial_conditions, t, args=(updated_params,))
    sol2 = odeint(love_dynamics, perturbed_initial, t, args=(updated_params,))

    # Calculate the divergence and Lyapunov exponent
    divergence = np.linalg.norm(sol2 - sol1, axis=1)
    divergence = np.ma.masked_where(divergence == 0, divergence)
    lyapunov = 1 / n * np.sum(np.log(divergence / perturbation_delta))

    return lyapunov

# Define parameters
params = [
    0.36,   # alpha1
    0.2,    # alpha2
    0.75,   # beta1
    10.66,  # beta2
    1,      # gamma1
    1,      # gamma2
    0.017,  # A1
    0.1,    # A2
    1,      # x_star2 (adjust as needed)
    0.1,    # epsilon
    100,    # mu
    1       # delta (adjust as needed)
]

initial_conditions = [0, 0, 0]  # x1, x2, z2
omega = 2 * np.pi / 52  # Define the value of omega

# Main execution block
if __name__ == '__main__':
    a1_values = np.linspace(0.05, 0.19, 100)
    delta_values = np.linspace(0, 1, 100)

    param_combinations = [(a1, delta) for a1 in a1_values for delta in delta_values]

    results = []
    for a1, delta in param_combinations:
        # Update delta in params for each iteration
        params[11] = delta  # Assuming delta is at index 11 in the params list
        lle = largest_lyapunov_exponent(initial_conditions, a1, delta, params, omega)
        results.append(lle)

    LLE_values = np.array(results).reshape(len(a1_values), len(delta_values))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(LLE_values, cmap='seismic', vmin=-2, vmax=2, extent=[delta_values.min(), delta_values.max(), a1_values.min(), a1_values.max()], aspect='auto', origin='lower')
    plt.colorbar(label='Largest Lyapunov Exponent')
    plt.xlabel('Delta')
    plt.ylabel('A1')
    plt.title('Heatmap of Lyapunov Exponent for different A1 and Delta')
    plt.show()
