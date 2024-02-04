"""
The :mod:'love.lyapunov_exponent' module implements The Lyapunov exponent 
is computed as the average of the logarithm of the ratio of the distances 
between two nearby trajectories in phase space
"""

"""Modeling the love story in Gone with the Wind"""

# Author: Yehui He <yehui.he@hotmail.com>


# License: MIT

import numpy as np
from scipy.integrate import odeint


def lyapunov_exponent(data, eps=0.0001):
    """
    Lyapunov characteristic exponent of a dynamical system.

    A quantity that characterizes the rate of separation of infinitesimally close 
    trajectories.

    Parameters
    ----------
    data : series, np.array 
        Data for the calculation.
    eps : float
        Threshold. (default 0.0001)

    Returns
    -------
    res : float
        The value of the lyapunovs exponent

    References
    ----------
    .. [1] N.V. Kuznetsov; G.A. Leonov (2005). "On stability by the 
           first approximation for discrete systems". Proceedings. 2005 International 
           Conference Physics and Control, 2005. Vol. Proceedings Volume 2005. pp. 596–599.
    
    .. [2] G.A. Leonov; N.V. Kuznetsov (2007). "Time-Varying Linearization and 
           the Perron effects". International Journal of Bifurcation and Chaos. 17 (4): 1079–1107.
    """

    N = len(data)
    lyapunovs = []
    for i in range(N):
        for j in range(i + 1, N):
            if np.abs(data[i] - data[j]) < eps:
                for k in range(1, min(N - i, N - j)):
                    d0 = np.abs(data[i] - data[j])
                    dn = np.abs(data[i + k] - data[j + k])
                    lyapunovs.append(np.log(dn / d0))

    return np.mean(lyapunovs)

def largest_lyapunov_exponent(love_dynamics, initial_conditions, A1, epsilon, params, omega, delta=0.0001, T=208, dt=0.02):
    t = np.arange(0, T, dt)
    n = len(t)
    
    perturbed_initial = initial_conditions + np.random.normal(0, delta, len(initial_conditions))

    updated_params = params.copy()
    updated_params[8] = A1 
    
    sol1 = odeint(love_dynamics, initial_conditions, t, args=(updated_params, epsilon, omega))
    sol2 = odeint(love_dynamics, perturbed_initial, t, args=(updated_params, epsilon, omega))

    divergence = np.linalg.norm(sol2 - sol1, axis=1)
    divergence = np.ma.masked_where(divergence == 0, divergence)
    lyapunov = 1/n * np.sum(np.log(divergence/delta))

    return lyapunov