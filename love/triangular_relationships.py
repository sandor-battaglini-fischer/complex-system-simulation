import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import time
from multiprocessing import Pool

# Names for the central partner and the two lovers
central_partner = "Woman"
lover1 = "Husband"
lover2 = "Lover"

# Kathe's reaction function to love from Jules
def RL12(x21, tauI12, sigmaL12, sigmaI12, beta12):
    """Kathe's reaction function to love from Jules

    Args:
        x21 (float): love from Jules
        tauI12 (float): insecurity threshold for Kathe's reaction to Jules' love
        sigmaL12 (float): sensitivity of reaction to love for Kathe to Jules
        sigmaI12 (float): sensitivity of insecurity for Kathe to Jules
        beta12 (float): reaction coefficient to love for Kathe to Jules love(years^-1)

    Returns:
        float: Kathe's reaction function to love from Jules
    """
    if x21 >= tauI12:
        return beta12 * x21 / (1 + x21/sigmaL12) * (1 - ((x21 - tauI12) / sigmaI12)**2) / (1 + ((x21 - tauI12) / sigmaI12)**2)
    else:
        return beta12 * x21 / (1 + x21/sigmaL12)

# Synergism function of Kathe (j=2,3)
def S(x1j, tau_S, sigmaS, s):
    """Synergism function of Kathe (j=2,3)

    Args:
        x1j (float): love from Jules or Jim
        tau_S (float): synergism threshold for Kathe
        sigmaS (float): sensitivity of synergism for Kathe
        s (float): synergism coefficient for Kathe

    Returns:
        float: Synergism function of Kathe
    """
    if x1j >= tau_S:
        return s*((x1j - tau_S) / sigmaS)**2 / (1 + ((x1j - tau_S) / sigmaS)**2)
    else:
        return 0

# Platonicity function as defined by Jules
def P(x21, tauP, p, sigmaP):
    """Platonicity function as defined by Jules

    Args:
        x21 (float): love from Jules
        tauP (float): platonicity threshold for Jules
        p (float): maximum platonicity for Jules
        sigmaP (float): sensitivity of platonicity for Jules

    Returns:
        float: Platonicity function as defined by Jules
    """
    if x21 >= tauP:
        return p*((x21 - tauP) / sigmaP)**2 / (1 + ((x21 - tauP) / sigmaP)**2)
    else:
        return 0

# Jim's reaction function to love from Kathe
def RL31(x13, tauI31, beta31, sigmaL31, sigmaI31):
    """Jim's reaction function to love from Kathe

    Args:
        x13 (float): love from Kathe
        tauI31 (float): insecurity threshold for Jim's reaction to love
        beta31 (float): reaction coefficient to love for Jim
        sigmaL31 (float): sensitivity of reaction to love for Jim
        sigmaI31 (float): sensitivity of insecurity for Jim

    Returns:
        float: Jim's reaction function to love from Kathe
    """
    if x13 >= tauI31:
        return beta31 * x13 / (1 + x13/sigmaL31) * (1 - ((x13 - tauI31) / sigmaI31)**2) / (1 + ((x13 - tauI31) / sigmaI31)**2)
    else:
        return beta31 * x13 / (1 + x13/sigmaL31)

def love_dynamics(y, t, params):
    """Love dynamics of Kathe, Jules, and Jim

    Args:
        y (list): list of data
        t (list): list of time data
        params (list): list of parameters

    Returns:
        list: list of love dynamics
    """
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


def default_triangular_plot():
    """Plot the default triangular relationship
    """

    initial_conditions = [0, 0, 0, 0]
    t = np.linspace(0, 20, 1000) 
    solution = odeint(love_dynamics, initial_conditions, t, args=(params,))

    # Extract solutions
    x12, x13, x21, x31 = solution.T

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Dynamics between central partner and lover1
    axs[0, 0].plot(t, x12, label=f'{central_partner} to {lover1}', color='tab:pink')
    axs[0, 0].plot(t, x21, label=f'{lover1} to {central_partner}', color='tab:blue')
    axs[0, 0].set_xlabel('Time (years)')
    axs[0, 0].set_ylabel('Feelings')
    axs[0, 0].set_title(f'{central_partner}-{lover1} Dynamics')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Phase diagram between central partner and lover1
    axs[0, 1].plot(x12, x21)
    axs[0, 1].set_xlabel(f'Feelings of {central_partner} towards {lover1}')
    axs[0, 1].set_ylabel(f'Feelings of {lover1} towards {central_partner}')
    axs[0, 1].set_title('Phase Diagram')
    axs[0, 1].grid(True)

    # Dynamics between central partner and lover2
    axs[1, 0].plot(t, x13, label=f'{central_partner} to {lover2}', color='tab:pink')
    axs[1, 0].plot(t, x31, label=f'{lover2} to {central_partner}', color='tab:green')
    axs[1, 0].set_xlabel('Time (years)')
    axs[1, 0].set_ylabel('Feelings')
    axs[1, 0].set_title(f'{central_partner}-{lover2} Dynamics')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Phase diagram between central partner and lover2
    axs[1, 1].plot(x13, x31)
    axs[1, 1].set_xlabel(f'Feelings of {central_partner} towards {lover2}')
    axs[1, 1].set_ylabel(f'Feelings of {lover2} towards {central_partner}')
    axs[1, 1].set_title('Phase Diagram')
    axs[1, 1].grid(True)

    # Imbalance plot
    imbalance = x12 - x13
    zero_crossings = np.where(np.diff(np.sign(imbalance)))[0]
    times_of_zero_crossings = t[zero_crossings] + \
                            (t[zero_crossings + 1] - t[zero_crossings]) * \
                            (0 - imbalance[zero_crossings]) / \
                            (imbalance[zero_crossings + 1] - imbalance[zero_crossings])

    axs[2, 0].plot(t, imbalance, color='tab:pink')
    axs[2, 0].scatter(times_of_zero_crossings, np.zeros_like(times_of_zero_crossings), color='red', zorder=5)
    axs[2, 0].fill_between(t, imbalance, where=(imbalance > 0), color='tab:blue', alpha=0.3)
    axs[2, 0].fill_between(t, imbalance, where=(imbalance < 0), color='tab:green', alpha=0.3)
    axs[2, 0].axhline(y=0, color='black', linestyle='--')
    axs[2, 0].set_xlabel('Time (years)')
    axs[2, 0].set_ylabel(f'Imbalance of {central_partner}\'s feelings \n towards {lover1} (>0) and {lover2} (<0)')
    axs[2, 0].set_title('Imbalance Over Time')
    axs[2, 0].grid(True)

    number_of_zeros = len(times_of_zero_crossings) - 2
    axs[2, 1].text(0.5, 0.5, f'Partner switching: {number_of_zeros}', 
                horizontalalignment='center', 
                verticalalignment='center', 
                fontsize=16, 
                transform=axs[2, 1].transAxes)
    axs[2, 1].axis('off')

    fig.tight_layout()
    plt.show()
    
