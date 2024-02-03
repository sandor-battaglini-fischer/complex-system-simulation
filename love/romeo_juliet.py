"""General linear sytemn"""

# Author: Yehui He <yehui.he@hotmail.com>

# License: MIT


__all__ = ["romeo_juliee"]
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def romeo_juliet(z, t, a, b, c, d):
    """
    General linear system.

    Modeling love affairs with system of ordinary first order homogeneous differential 
    equations with constant coefficients

    Parameters
    ----------
    z : list of length 2 
        Romeo’s love / hate for Juliet at time t.
        Juliet’s love / hate for Romeo at time t.
    t : float
        initial time
    a : float
        Response to love for Romeo.
    b : float
        Appeal of Juliet.
    c : float
        Appeal of Romeo.
    d : float
        Response to love for Juliet

    Returns
    -------
    res : float
        The value of the vector field

    References
    ----------

    .. [1] Strogatz, S. H. (1988) Love affairs and differential equations. Math. Magazine
           61, 35.
    """

    R, J = z

    res = [a*R + b*J, c*R + d*J]

    return res 

def plot_phase_portrait(func, params):
    """plots phase portrait

    Args:
        func (function): function 
        params (array): parameters to use with func
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
            U[i,j] = romeo_juliet([x, y], t, *params)[0]
            V[i,j] = romeo_juliet([x, y], t, *params)[1]    
    # plot
    fig, ax = plt.subplots()
    
    ax.quiver(xv, yv, U, V, color="C0", 
              angles='xy',
              scale_units='xy', scale=5, width=.015)
    
    ax.set(xlabel='R', ylabel='J', xlim=(-5, 5), ylim=(-5, 5))
    
    # for y20 in [0, 0.5, 1, 1.5, 2, 2.5]:
    #     tspan = np.linspace(0, 50, 200)
    #     y0 = [0.0, y20]
    #     ys = odeint(f, y0, tspan)
    #     plt.plot(ys[:,0], ys[:,1], 'b-') # path
    #     plt.plot([ys[0,0]], [ys[0,1]], 'o') # start
    #     plt.plot([ys[-1,0]], [ys[-1,1]], 's') # end
    
    for y20 in [2, 3, 4]:
        t_span = np.linspace(0, 15, 200)
        # sol = solve_ivp(romeo_juliet, [0, 200], [y0, y0], args=params, 
        #                 dense_output=True)
        y0 = [1.0, y20]
        sol = odeint(romeo_juliet, y0, t_span, args=params)
        plt.plot(sol[:,0], sol[:,1], 'r-') # path
        plt.plot([sol[0, 0]], [sol[0, 1]], 'o') # start
        plt.plot([sol[-1,0]], [sol[-1,1]], 's') # end
    
    plt.show()

