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


def _input_validation(data, eps):
    """
    Input validation for lyapunov_exponent
    """

    if not isinstance(data, list):
        raise TypeError(f"Arugment data is not of type list")

    if not isinstance(eps, float):
        raise TypeError(f"Arugment eps is not of type float")

    if eps < 0:
        raise ValueError(f"Augment eps is nonnegative")

def lyapunov_exponent(data, eps=0.0001):
    """
    Lyapunov characteristic exponent of a dynamical system.

    A quantity that characterizes the rate of separation of infinitesimally close 
    trajectories.

    Parameters
    ----------
    data : array-like
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

    _input_validation(data, eps)

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
<<<<<<< HEAD
    """
    Largest Lyapunov characteristic exponent of a dynamical system.

    A quantity that characterizes the rate of separation of infinitesimally close
    trajectories.

    Parameters
    ----------
    love_dynamics : callable
        love dynamics function.
    initial_conditions : list
        List of initial conditions for ODE

    Returns
    -------
    lyapunov : float
        The value of the lyapunovs exponent

    References
    ----------
    .. [1] N.V. Kuznetsov; G.A. Leonov (2005). "On stability by the
           first approximation for discrete systems". Proceedings. 2005 International
           Conference Physics and Control, 2005. Vol. Proceedings Volume 2005. pp. 596–599.

    .. [2] G.A. Leonov; N.V. Kuznetsov (2007). "Time-Varying Linearization and
           the Perron effects". International Journal of Bifurcation and Chaos. 17 (4): 1079–1107.
    """

    if not isinstance(epsilon, float):
        raise TypeError(f"Arugment epsilon is not of type float")

    if epsilon < 0:
        raise ValueError(f"Augment epsilon is nonnegative")
    
    if not isinstance(delta, float):
        raise TypeError(f"Arugment epsilon is not of type float")

    if delta < 0:
        raise ValueError(f"Augment epsilon is nonnegative")

=======
    """_summary_

    Args:
        love_dynamics (_type_): _description_
        initial_conditions (_type_): _description_
        A1 (_type_): _description_
        epsilon (_type_): _description_
        params (_type_): _description_
        omega (_type_): _description_
        delta (float, optional): _description_. Defaults to 0.0001.
        T (int, optional): _description_. Defaults to 208.
        dt (float, optional): _description_. Defaults to 0.02.

    Returns:
        _type_: _description_
    """
>>>>>>> 4c3d4f5d873e5833ad5c514cb3d8039299f17d64
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
