"""
The :mod:'love.lyapunov_exponent' module implements The Lyapunov exponent 
is computed as the average of the logarithm of the ratio of the distances 
between two nearby trajectories in phase space
"""

"""Modeling the love story in Gone with the Wind"""

# Author: Yehui He <yehui.he@hotmail.com>


# License: MIT

import numpy as np

def lyapunov_exponent(data, eps):
    """
    Lyapunov characteristic exponent of a dynamical system.

    A quantity that characterizes the rate of separation of infinitesimally close 
    trajectories.

    Parameters
    ----------
    data : series, np.array 
        Data for the calculation.
    eps : float
        Threshold.

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

