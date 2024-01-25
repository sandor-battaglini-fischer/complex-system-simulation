import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks


def love_dynamics(y, t, p, epsilon, omega):
    x1, x2 = y
    alpha1, alpha2, beta1, beta2, gamma1, gamma2, bA1, bA2, A1, A2, k1, k2, n1, n2, m1, m2, sigma1, sigma2 = p

    RL1 = beta1 * k1 * x2 * np.exp(-(k1 * x2)**n1)
    RL2 = beta2 * k2 * x1 * np.exp(-(k2 * x1)**n2)

    BA1 = x1**(2*m1) / (x1**(2*m1) + sigma1**(2*m1))
    BA2 = x2**(2*m2) / (x2**(2*m2) + sigma2**(2*m2))

    dx1dt = -alpha1 * x1 + RL1 + (1 + bA1 * BA1) * gamma1 * A2
    dx2dt = -alpha2 * x2 + RL2 + (1 + bA2 * BA2) * gamma2 * A1*(1+epsilon*np.sin(omega*t))

    return [dx1dt, dx2dt]


# Parameters
params = [
    0.36,   # alpha1    =   Forgetting coefficient 1 (decay rate of love of Romeo in absence of partner)
    0.2,    # alpha2    =   Forgetting coefficient 2
    0.75,   # beta1     =   Reactiveness to love of 2 on 1 (influence of the partner's love on an individual's feelings)
    10.66,  # beta2     =   Reactiveness to love of 1 on 2
    1,      # gamma1    =   Reactiveness to appeal of 2 on 1 (influence of the partner's appeal on an individual's feelings)
    1,      # gamma2    =   Reactiveness to appeal of 1 on 2
    2.9,    # bA1       =   Bias coefficient of Romeo (how much Romeo is biased towards their partner, > 0 for synergic, 0 for unbiased, < 0 for platonic)
    1,      # bA2       =   Bias coefficient of individual 2
    0.017,  # A1        =   Appeal of Romeo (how much Romeo is appealing to their partner)
    0.1,    # A2        =   Appeal of individual 2
    0.08,   # k1        =   Insecurity of Romeo (Peak of reaction function of 1 on 2, high k1 means they are annoyed by their partner's love earlier)
    1.5,    # k2        =   Insecurity of individual 2
    1,      # n1        =   Shape of reaction function of 1 on 2 (nonlinearity of reaction function of 1 on 2, sensitivity of the individuals' feelings to changes in their partner's feelings)
    4,      # n2        =   Shape of reaction function of 2 on 1
    4,      # m1        =   Shape of bias function of 1 (nonlinearity of bias function of 1, sensitivity of how the own feelings influence their perception of their partner's appeal)
    4,      # m2        =   Shape of bias function of 2
    1,      # sigma1    =   Saddle quantity of 1 (Trace of Jabobian of 1, threshold of when own feelings influence their perception of their partner's appeal. > 0 for stable, < 0 for unstable)
    1       # sigma2    =   Saddle quantity of 2
]


epsilon = 0.17
omega = 2 * np.pi / 52


initial_conditions = [0.895, 1.5]
t = np.linspace(0, 208, 10000)

solution = odeint(love_dynamics, initial_conditions, t, args=(params, epsilon, omega))


# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Time series
axs[0].plot(t, solution[:, 0], label='Romeo', color='tab:blue')
axs[0].plot(t, solution[:, 1], label='Juliet', color='tab:pink')
axs[0].set_xlabel('Time (weeks)')
axs[0].set_ylabel('Feelings')
axs[0].set_title('Dynamics of Romantic Relationship with Environmental Stress')
axs[0].legend()
axs[0].grid(True)


# Peaks
x1_data = solution[:, 0]

peaks, _ = find_peaks(x1_data)
peak_values = x1_data[peaks]

h_peaks = peak_values[:-1] 
hp1_peaks = peak_values[1:]


# PPP Diagram
axs[1].scatter(h_peaks, hp1_peaks, color='blue')
axs[1].set_xlabel(r'$x_{1, h}$ (Romeo Peak h)')
axs[1].set_ylabel(r'$x_{1, h+1}$ (Romeo Peak h+1)')
axs[1].set_title('Peak-to-Peak Plot (PPP) for Romeo')
axs[1].grid(True)

fig.tight_layout()
plt.show()

