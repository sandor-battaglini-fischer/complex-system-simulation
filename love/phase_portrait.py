"""Phase portrait plotting function"""

# Author: Yehui He <yehui.he@hotmail.com>
          Sándor Battaglini-Fischer <>

# License: MIT

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def plot_phase_portrait(func, params, init_conditions, xlabel, ylabel, xlim, ylim):
    """
    Phase portrait plotting function.

    Plotting phase portrait with corresponding functions

    Parameters
    ----------
    func : callable 
        2D Vector field represent mathematical modelling of the love affairs.
    params : tuple
        Extra arguments to pass to function.
    init_conditions : list
        Initial conditions for the systems.
    xlabel : str
        Label of x-axis.
    ylabel : str
        Label of y-axis.
    xlim : float
        Get or set the x limits of the current axes.
    ylim : float
        Get or set the y limits of the current axes.
    """

    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    xv, yv = np.meshgrid(x, y)
    
    t = 0
    
    U, V = np.zeros(xv.shape), np.zeros(yv.shape)
    
    nx, ny = xv.shape
    
    for i in range(nx):
        for j in range(ny):
            x = xv[i, j]
            y = yv[i, j]
            # yprime = f([x, y], t)
            U[i,j] = func([x, y], t, *params)[0]
            V[i,j] = func([x, y], t, *params)[1]    
    # plot
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    
    ax.quiver(xv, yv, U, V, color="C0", 
              angles='xy',
              scale_units='xy', scale=5, width=.015)
    
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    
    for y0 in init_conditions:
        t_span = np.linspace(0, 15, 200)
        sol = odeint(func, y0, t_span, args=params)
        plt.plot(sol[:,0], sol[:,1], 'r-') # path
        plt.plot([sol[0, 0]], [sol[0, 1]], 'o') # start
        plt.plot([sol[-1,0]], [sol[-1,1]], 's') # end
    
    plt.show()

