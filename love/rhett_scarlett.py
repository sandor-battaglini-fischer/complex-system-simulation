"""Modeling the love story in Gone with the Wind"""

# Author: Yehui He <yehui.he@hotmail.com>

# License: MIT


import math

__all__ = ["rhett_scarlett"]


def rhett_scarlett(z, t, A_s, A_r, k):
    """
    Modeling the love story in “Gone with the Wind”.

    Nonlinear system of differential equations for modelling
    Rhett and Scarlett's love story.

    Parameters
    ----------
    z : list of length 2
        Rhett’s love / hate for Scarlett at time t.
        Scarlett’s love / hate for Rhett at time t.
    t : float
        initial time
    A_s : float
        Scarlett's appeal to Rhett.
    A_r : float
        Rhett's appeal to Scarlett.
    k : float
        Oblivion (damping) strength
                            
    Returns             
    -------             
    res : float
        The value of the vector field
       
    References
    ----------

    .. [1] Rinaldi, S., Della Rossa, F., and Landi, P. (2013) A mathematical model of
           ‘‘Gone with the Wind.’’ Physica A 392, 3231.
    """

    R, S = z

    res = [-R + A_s + k*S*math.exp(-S), -S + A_r + k*R*math.exp(-R)]

    return res 

