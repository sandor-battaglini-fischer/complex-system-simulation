"""
Love Multiple Network
"""

# Author: Victor Piaskowski <victor.piaskowski@student.uva.nl>

# License: MIT

import numpy as np
import random
from scipy.integrate import solve_ivp

__all__ = ["calculateODE", "calculateODEMatrixVector", "random_starting_matrix", "random_starting_vector", "random_initial_vector", "randomize_parameter_network"]

def calculateODE(axx,axy,bxx,bxy,cxx,cxy,dxx,dxy,fxy,gxy,ayy,ayx,byy,byx,cyy,cyx,dyy,dyx,fyx,gyx,xi0,yi0,xp0,yp0): 
    """Calculates an ODE based on a list of parameters

    Args:
        axx (float): Forgetting coefficient of the intimacy of Xena to Yorgo.
        axy (float): If Yorgo’s intimacy increases, Xena’s will decrease, and if it decreases, it will increase. If Xena’s partner shows closeness/interest to her, Xena gradually loses her sense of intimacy.
        bxx (float): If Xena’s passion increases, her sense of intimacy increases, and if it decreases, it decreases. She is intimate with someone she is passionate about. She might just want to fall in love.
        bxy (float): As Yorgo’s passion for Xena increase, Xena’s closeness to Yorgo decreases. When she realizes that Yorgo is not in love, Xena increases her intimacy. Maybe she doesn’t want someone in love with her.
        cxx (float): Her passion increase when Xena feels close. Men with whom she does not feel close are not attractive, but men with whom she feels sincere can be attractive.
        cxy (float): Intimate men are very attractive to Xena. Her passion for men who do not behave closely is significantly reduced.
        dxx (float): Forgetting coefficient of the passion of Xena for Yorgo.
        dxy (float): As Yorgo’s passion grows, so does Xena’s. A man who acts romantic may attract her.
        fxy (float): Xena’s impression of intimacy or friendship with Yorgo. Xena finds Yorgo intimate and friendly. She enjoys being friends and spending time with him.
        gxy (float): Xena’s impression of glamorousness or attractiveness about Yorgo. Xena does not find Yorgo romantically or sexually attractive.
        ayy (float): Forgetting coefficient of the intimacy of Yorgo to Xena.
        ayx (float): If Yorgo’s intimacy increases, Xena’s will decrease, if it decreases, it will increase. If Yorgo’s partner shows intimacy/interest to him, Yorgo increases his sense of intimacy.
        byy (float): If Yorgo’s passion increases, his sense of intimacy decreases, and if it decreases, it increases. He is intimate with someone he is not passionate about. He might just want not to fall in love.
        byx (float): As Xena’s passion for Yorgo increases, Yorgo’s intimacy with Xena increases. When he realizes that Xena is not in love, Yorgo decreases his intimacy. Maybe he wants someone in love with her.
        cyy (float): His passion decreases when Yorgo feels close. Women with whom he does not feel close are attractive, but women with whom he feels intimacy are not attractive.
        cyx (float): Intimate women are not attractive to Yorgo. His passion for women who are close to him weakens a little.
        dyy (float): Forgetting coefficient of the passion of Yorgo to Xena.
        dyx (float): As Xena’s passion increases, Yorgo’s decreases. A romantic woman does not attract him.
        fyx (float): Yorgo’s impression of intimacy or friendship with Xena. Yorgo found Xena neither sympathetic nor antisympathetic.
        gyx (float): Yorgo’s impression of glamorousness or attractiveness about Xena. Yorgo finds Xena attractive. He desires her romantically and sexually.
        xi0 (float): intial value of Xena's intimacy for Yorgo
        yi0 (float): intial value of Yorgo's intimacy for Xena
        xp0 (float): intial value of Xena's passion for Yorgo
        yp0 (float): intial value of Yorgo's passion for Xena

    Returns:
        array: solution to initial value problem
    """
    A = np.array([[axx, axy, bxx, bxy], [ayx, ayy, byx, byy], [cxx, cxy, dxx, dxy], [cyx, cyy, dyx, dyy]])
    B = np.array([fxy, fyx, gxy, gyx])
    vdp1 = lambda T, x: A.dot(np.array([x[0], x[1], x[2], x[3]])) + B
    sol = solve_ivp (vdp1, [0, 4], np.array([xi0, yi0, xp0, yp0]), max_step=0.1)
    return sol

def calculateODEMatrixVector(A, B, x): 
    """calculated IVP of matrix A * vector B + vector C. Same exact as calculateODE, just already in a matrix form

    Args:
        A (matrix): matrix of coefficients
        B (vector): vector of coefficients
        x (vector): vector of coefficients

    Returns:
        array: solution to initial value problem
    """
    vdp1 = lambda T, x: A.dot(np.array([x[0], x[1], x[2], x[3]])) + B
    sol = solve_ivp (vdp1, [0, 30], np.array([x[0], x[1], x[2], x[3]]), max_step=0.1)
    return sol

def random_starting_matrix():
    """generate a random 4x4 matrix

    Returns:
        4x4 numpy matrix: a 4x4 matrix with vlaues ranging between -1 and 1
    """
    A = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            A[i, j] = round(random.uniform(-1, 1), 2)
    return A

def random_starting_vector():
    """generate a random vector of length 4 with vlaues between -1 and 1

    Returns:
        vector: vector of length 4 with values rnaging between -1 and 1
    """
    A = np.ones((4))
    for i in range(4):
        A[i] = round(random.uniform(-1, 1), 2)
    return A

def random_initial_vector():
    """generates a random starting vector of length 4 with vlaues between -1_000_000 and 1_000_000

    Returns:
        vector: vector of length 4 with values rnaging between -1_000_000 and 1_000_000
    """
    A = np.ones((4))
    for i in range(4):
        A[i] = round(random.uniform(-1_000_000, 1_000_000), 2)
    return A

def randomize_parameter_network(G, number_of_nodes):
    """randomiz parameters of a network

    Args:
        G (Graph): graph of the network
        number_of_nodes (int): number of nodes in G

    Returns:
        Graph: graph of the network with new parameters
    """
    for i in range(number_of_nodes):
        for j in range(i+1, number_of_nodes):
            G[i][j]['parameters'] = [random_starting_matrix(), random_starting_vector(), random_initial_vector()]
            calculatedODE = calculateODEMatrixVector(G[i][j]['parameters'][0], G[i][j]['parameters'][1], G[i][j]['parameters'][2])
            G[i][j]["data"] = calculatedODE.y
            G[i][j]["t"] = calculatedODE.t
    return G