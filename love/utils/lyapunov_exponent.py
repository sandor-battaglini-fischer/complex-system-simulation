"""
The :mod:'love.lyapunov_exponent' module implements The Lyapunov exponent 
is computed as the average of the logarithm of the ratio of the distances 
between two nearby trajectories in phase space
"""

import numpy as np

def lyapunov_exponent(data):
    N = len(data)
    eps = 0.0001
    lyapunovs = []
    for i in range(N):
        for j in range(i + 1, N):
            if np.abs(data[i] - data[j]) < eps:
                for k in range(1, min(N - i, N - j)):
                    d0 = np.abs(data[i] - data[j])
                    dn = np.abs(data[i + k] - data[j + k])
                    lyapunovs.append(np.log(dn / d0))
    return np.mean(lyapunovs)

