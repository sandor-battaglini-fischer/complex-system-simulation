import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

""" 
Model of Love Dynamics 
from chapter 14 and 15 of the book.

Model of Kathe-Jules-Jim triangular love dynamics.

"""

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
def RL31(x13, tauI31, beta3, sigmaL31, sigmaI31):
    if x13 >= tauI31:
        return beta3 * x13 / (1 + x13/sigmaL31) * (1 - ((x13 - tauI31) / sigmaI31)**2) / (1 + ((x13 - tauI31) / sigmaI31)**2)
    else:
        return beta3 * x13 / (1 + x13/sigmaL31)


def love_dynamics(y, t, params):
    x12, x13, x21, x31 = y
    alpha1, alpha2, alpha3, beta2, beta12, beta13, beta3, gamma1, gamma2, gamma3, epsilon, delta, A1, A2, A3, tauI12, sigmaL12, sigmaI12, beta12, tau_S, sigmaS, tauP, p, sigmaP, tauI31, sigmaL31, sigmaI31, s = params

    dx12dt = -alpha1 * np.exp(epsilon * (x13 - x12)) * x12 + RL12(x21, tauI12, sigmaL12, sigmaI12, beta12) + (1 + S(x12, tau_S, sigmaS, s)) * gamma1 * A2
    dx13dt = -alpha1 * np.exp(epsilon * (x12 - x13)) * x13 + beta13 * x31 + (1 + S(x13, tau_S, sigmaS, s)) * gamma1 * A3
    dx21dt = -alpha2 * x21 + beta2 * x12 * np.exp(delta * (x13 - x12)) + (1 - P(x21, tauP, p, sigmaP)) * gamma2 * A1
    dx31dt = -alpha3 * x31 + RL31(x13, tauI31, beta3, sigmaL31, sigmaI31) * np.exp(delta * (x13 - x12)) + gamma3 * A1

    return [dx12dt, dx13dt, dx21dt, dx31dt]

# Parameters w/ extensive comments
params = [
    2,    # alpha1: forgetting coefficient for Kathe
    1,    # alpha2: forgetting coefficient for Jules
    2,    # alpha3: forgetting coefficient for Jim
    1,    # beta2: reaction coefficient to love for Jules
    8,    # beta12: reaction coefficient to love for Kathe to Jules love
    1,    # beta13: reaction coefficient to love for Kathe to Jim's love
    2,    # beta3: reaction coefficient to love for Jim
    1,    # gamma1: reaction coefficient to appeal for Kathe
    0.5,  # gamma2: reaction coefficient to appeal for Jules
    1,    # gamma3: reaction coefficient to appeal for Jim
    0.0062,   # epsilon: sensitivity of reaction to love for Kathe
    0.0285,    # delta: sensitivity of reaction to love for Jules and Jim
    20,   # A1: appeal of Kathe
    4,    # A2: appeal of Jules
    5,    # A3: appeal of Jim
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
t = np.linspace(0, 20, 1000) 
solution = odeint(love_dynamics, initial_conditions, t, args=(params,))

# Extract solutions
x12, x13, x21, x31 = solution.T

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Kathe-Jules dynamics
axs[0, 0].plot(t, x12, label='Kathe to Jules', color='tab:pink')
axs[0, 0].plot(t, x21, label='Jules to Kathe', color='tab:blue')
axs[0, 0].set_xlabel('Time (years)')
axs[0, 0].set_ylabel('Feelings')
axs[0, 0].set_title('Kathe-Jules Dynamics')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Kathe-Jules phase diagram
axs[0, 1].plot(x12, x21)
axs[0, 1].set_xlabel('Feelings of Kathe towards Jules')
axs[0, 1].set_ylabel('Feelings of Jules towards Kathe')
axs[0, 1].set_title('Phase Diagram')
axs[0, 1].grid(True)

# Kathe-Jim dynamics
axs[1, 0].plot(t, x13, label='Kathe to Jim', color='tab:pink')
axs[1, 0].plot(t, x31, label='Jim to Kathe', color='tab:green')
axs[1, 0].set_xlabel('Time (years)')
axs[1, 0].set_ylabel('Feelings')
axs[1, 0].set_title('Kathe-Jim Dynamics')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Kathe-Jim phase diagram
axs[1, 1].plot(x13, x31)
axs[1, 1].set_xlabel('Feelings of Kathe towards Jim')
axs[1, 1].set_ylabel('Feelings of Jim towards Kathe')
axs[1, 1].set_title('Phase Diagram')
axs[1, 1].grid(True)

#Imbalance
imbalance = x12 - x13
zero_crossings = np.where(np.diff(np.sign(imbalance)))[0]
times_of_zero_crossings = t[zero_crossings] + \
                          (t[zero_crossings + 1] - t[zero_crossings]) * \
                          (0 - imbalance[zero_crossings]) / \
                          (imbalance[zero_crossings + 1] - imbalance[zero_crossings])

axs[2, 0].plot(t, imbalance)
axs[2, 0].scatter(times_of_zero_crossings, np.zeros_like(times_of_zero_crossings), color='red', zorder=5)
axs[2, 0].axhline(y=0, color='black', linestyle='--')
axs[2, 0].set_xlabel('Time (years)')
axs[2, 0].set_ylabel(r'Imbalance $x_{12} - x_{13}$')
axs[2, 0].set_title('Imbalance Over Time')
axs[2, 0].grid(True)

number_of_zeros = len(times_of_zero_crossings)
axs[2, 1].text(0.5, 0.5, f'Partner switching: {number_of_zeros}', 
               horizontalalignment='center', 
               verticalalignment='center', 
               fontsize=16, 
               transform=axs[2, 1].transAxes)
axs[2, 1].axis('off')

fig.tight_layout()

plt.show()

