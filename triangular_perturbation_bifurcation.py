import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

# Names for the central partner and the two lovers
central_partner = "Woman"
lover1 = "Husband"
lover2 = "Lover"

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

def perturb_initial_conditions(initial_conditions, perturbations):
    """
    Perturb specified initial conditions.
    
    :param initial_conditions: List of initial conditions.
    :param perturbations: Dictionary with index as key and perturbation strength as value.
    :return: Perturbed initial conditions.
    """
    perturbed_conditions = initial_conditions.copy()
    for index, strength in perturbations.items():
        perturbed_conditions[index] += strength * perturbed_conditions[index]
    return perturbed_conditions


def perturb_parameters(params, perturbations):
    """
    Perturb specified parameters.
    
    :param params: List of parameters.
    :param perturbations: Dictionary with index as key and perturbation strength as value.
    :return: Perturbed parameters.
    """
    perturbed_params = params.copy()
    for index, strength in perturbations.items():
        perturbed_params[index] += strength
    return perturbed_params




def plot_love_dynamics(t, solution, perturbed_solution):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    x12, x13, x21, x31 = solution.T
    px12, px13, px21, px31 = perturbed_solution.T

    # Dynamics between central partner and lover1
    axs[0, 0].plot(t, x12, label=f'Original {central_partner} to {lover1}', color='tab:pink')
    axs[0, 0].plot(t, x21, label=f'Original {lover1} to {central_partner}', color='tab:blue')
    axs[0, 0].plot(t, px12, '--', label=f'Perturbed {central_partner} to {lover1}', color='tab:pink')
    axs[0, 0].plot(t, px21, '--', label=f'Perturbed {lover1} to {central_partner}', color='tab:blue')
    axs[0, 0].set_xlabel('Time (years)')
    axs[0, 0].set_ylabel('Feelings')
    axs[0, 0].set_title(f'{central_partner}-{lover1} Dynamics')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Phase diagram between central partner and lover1
    axs[0, 1].plot(x12, x21, label='Original', color='tab:blue')
    axs[0, 1].plot(px12, px21, '--', label='Perturbed', color='tab:blue', alpha=0.6)
    axs[0, 1].set_xlabel(f'Feelings of {central_partner} towards {lover1}')
    axs[0, 1].set_ylabel(f'Feelings of {lover1} towards {central_partner}')
    axs[0, 1].set_title('Phase Diagram')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Dynamics between central partner and lover2
    axs[1, 0].plot(t, x13, label=f'Original {central_partner} to {lover2}', color='tab:pink')
    axs[1, 0].plot(t, x31, label=f'Original {lover2} to {central_partner}', color='tab:green')
    axs[1, 0].plot(t, px13, '--', label=f'Perturbed {central_partner} to {lover2}', color='tab:pink')
    axs[1, 0].plot(t, px31, '--', label=f'Perturbed {lover2} to {central_partner}', color='tab:green')
    axs[1, 0].set_xlabel('Time (years)')
    axs[1, 0].set_ylabel('Feelings')
    axs[1, 0].set_title(f'{central_partner}-{lover2} Dynamics')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Phase diagram between central partner and lover2
    axs[1, 1].plot(x13, x31, label='Original', color='tab:blue')
    axs[1, 1].plot(px13, px31, '--', label='Perturbed', color='tab:blue', alpha=0.6)
    axs[1, 1].set_xlabel(f'Feelings of {central_partner} towards {lover2}')
    axs[1, 1].set_ylabel(f'Feelings of {lover2} towards {central_partner}')
    axs[1, 1].set_title('Phase Diagram')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Imbalance plot
    imbalance = x12 - x13
    perturbed_imbalance = px12 - px13
    zero_crossings = np.where(np.diff(np.sign(imbalance)))[0]
    zero_crossings_perturbed = np.where(np.diff(np.sign(perturbed_imbalance)))[0]
    times_of_zero_crossings = t[zero_crossings] + \
                            (t[zero_crossings + 1] - t[zero_crossings]) * \
                            (0 - imbalance[zero_crossings]) / \
                            (imbalance[zero_crossings + 1] - imbalance[zero_crossings])
    times_of_zero_crossings_perturbed = t[zero_crossings_perturbed] + \
                            (t[zero_crossings_perturbed + 1] - t[zero_crossings_perturbed]) * \
                            (0 - perturbed_imbalance[zero_crossings_perturbed]) / \
                            (perturbed_imbalance[zero_crossings_perturbed + 1] - perturbed_imbalance[zero_crossings_perturbed])

    axs[2, 0].plot(t, imbalance, color='tab:pink')
    axs[2, 0].plot(t, perturbed_imbalance, '--', color='tab:pink')
    axs[2, 0].scatter(times_of_zero_crossings, np.zeros_like(times_of_zero_crossings), color='tab:red', zorder=5)
    axs[2, 0].scatter(times_of_zero_crossings_perturbed, np.zeros_like(times_of_zero_crossings_perturbed), color='tab:red', alpha=0.3, zorder=5)
    axs[2, 0].fill_between(t, imbalance, where=(imbalance > 0), color='tab:blue', alpha=0.3)
    axs[2, 0].fill_between(t, imbalance, where=(imbalance < 0), color='tab:green', alpha=0.3)
    axs[2, 0].fill_between(t, perturbed_imbalance, where=(perturbed_imbalance > 0), color='tab:blue', alpha=0.1)
    axs[2, 0].fill_between(t, perturbed_imbalance, where=(perturbed_imbalance < 0), color='tab:green', alpha=0.1)
    axs[2, 0].axhline(y=0, color='black', linestyle='--')
    axs[2, 0].set_xlabel('Time (years)')
    axs[2, 0].set_ylabel(f'Imbalance of {central_partner}\'s feelings \n towards {lover1} (>0) and {lover2} (<0)')
    axs[2, 0].set_title('Imbalance Over Time')
    axs[2, 0].grid(True)

    number_of_zeros = max(len(times_of_zero_crossings) - 2, 0)
    number_of_zeros_perturbed = max(len(times_of_zero_crossings_perturbed) - 2, 0)
    axs[2, 1].text(0.5, 0.5, f'Partner switching: {number_of_zeros}, \n Perturbed partner switching: {number_of_zeros_perturbed}',
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10,
                   transform=axs[2, 1].transAxes)
    axs[2, 1].axis('off')

    fig.tight_layout()
    # plt.show()
    
def count_partner_switchings(time_series):
    # Count the number of times the time series crosses zero
    zero_crossings = np.where(np.diff(np.sign(time_series)))[0]
    return len(zero_crossings)

def simulate_for_a3_range(a3_range, initial_conditions, params_template, t):
    switchings = []

    for a3 in a3_range:
        params = params_template.copy()
        params[14] = a3  # Set the A3 value
        solution = odeint(love_dynamics, initial_conditions, t, args=(params,))
        imbalance = solution[:, 0] - solution[:, 1]  # Assuming x12 - x13 is the imbalance
        num_switchings = count_partner_switchings(imbalance)
        switchings.append(num_switchings)

    return switchings

def plot_bifurcation_diagram(a3_range, switchings):
    plt.figure(figsize=(10, 6))
    plt.plot(a3_range, switchings, 'b-')
    plt.xlabel('Appeal of Jim (A3)')
    plt.ylabel('Number of Partner Switchings')
    plt.title('Bifurcation Diagram')
    plt.grid(True)
    plt.show()
    
def calculate_integral_balance(t, solution):
    # Assuming x21 represents Jules' feelings towards the central partner
    # and x31 represents Jim's feelings towards the central partner
    area_jules = np.trapz(solution[:, 2], t)  # Integral for Jules
    area_jim = np.trapz(solution[:, 3], t)    # Integral for Jim
    integral_balance = area_jules - area_jim
    return integral_balance

def generate_heatmap_data(a3_range, a1_range, initial_conditions, params_template, t):
    heatmap_data = np.zeros((len(a3_range), len(a1_range)))

    for i in tqdm(range(len(a3_range)), desc='A1 Progress'):
        for j in range(len(a1_range)):
            a3 = a3_range[i]
            a1 = a1_range[j]
            params = params_template.copy()
            params[14] = a3  # A1: appeal of Kathe
            params[12] = a1  # A3: appeal of Jim
            solution = odeint(love_dynamics, initial_conditions, t, args=(params,))
            integral_balance = calculate_integral_balance(t, solution)
            heatmap_data[i, j] = integral_balance

    return heatmap_data

def plot_integral_balance_heatmap(a1_range, a3_range, heatmap_data):
    # Normalize the heatmap_data around zero
    max_abs_value = np.max(np.abs(heatmap_data))
    norm_heatmap_data = heatmap_data / max_abs_value  # Normalized to [-1, 1]

    plt.figure(figsize=(10, 8))
    plt.imshow(norm_heatmap_data, extent=[a1_range[0], a1_range[-1], a3_range[0], a3_range[-1]],
               aspect='auto', cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label='Normalized Integral Balance between Jules (+) and Jim (-)')
    plt.xlabel('Appeal of Jim (A3)')
    plt.ylabel('Appeal of Kathe (A1)')
    plt.title('Heatmap of Normalized Integral Balance')
    plt.grid(False)
    plt.show()
    
def calculate_gradient(heatmap_data, a1_range, a3_range):
    # Compute the gradient along both axes
    gradient_a3 = np.gradient(heatmap_data, axis=0)
    gradient_a1 = np.gradient(heatmap_data, axis=1)

    # Calculate the magnitude of the gradient vector
    gradient_magnitude = np.sqrt(gradient_a3**2 + gradient_a1**2)

    return gradient_magnitude



if __name__ == '__main__':
    t = np.linspace(0, 50, 1000)
    initial_conditions = [0, 0, 0, 0]
    a1_range = np.linspace(0, 20, 100) 
    a3_range = np.linspace(0, 20, 100)

    # # Simulate for a range of A3 values
    # switchings = simulate_for_a3_range(a3_range, initial_conditions, params, t)

    # # Plot the bifurcation diagram
    # plot_bifurcation_diagram(a3_range, switchings)
    

    # Generate heatmap data
    heatmap_data = generate_heatmap_data(a3_range, a1_range, initial_conditions, params, t)

    # Plot the heatmap
    plot_integral_balance_heatmap(a3_range, a1_range, heatmap_data)
    
    # gradient_magnitude = calculate_gradient(heatmap_data, a3_range, a1_range)

    # # You can plot the gradient magnitude as a heatmap to visualize where the largest changes are occurring
    # plt.figure(figsize=(10, 8))
    # plt.imshow(gradient_magnitude, extent=[a3_range[0], a3_range[-1], a1_range[0], a1_range[-1]],
    #         aspect='auto', cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Gradient Magnitude of Integral Balance')
    # plt.xlabel('Appeal of Jim (A3)')
    # plt.ylabel('Appeal of Kathe (A1)')
    # plt.title('Gradient Magnitude Heatmap')
    # plt.grid(False)
    # plt.show()



