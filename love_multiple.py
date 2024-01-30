import numpy as np
import random
from scipy.integrate import solve_ivp
def random_starting_matrix():
    A = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            A[i, j] = round(random.uniform(-1, 1), 2)
    return A

def random_starting_vector():
    A = np.ones((4))
    for i in range(4):
        A[i] = round(random.uniform(-1, 1), 2)
    return A

def xy_starting_vector():
    A = np.zeros((4))
    return A

def calculateODE(axx,axy,bxx,bxy,cxx,cxy,dxx,dxy,fxy,gxy,ayy,ayx,byy,byx,cyy,cyx,dyy,dyx,fyx,gyx,xi0,yi0,xp0,yp0): 
    A = np.array([[axx, axy, bxx, bxy], [ayx, ayy, byx, byy], [cxx, cxy, dxx, dxy], [cyx, cyy, dyx, dyy]])
    B = np.array([fxy, fyx, gxy, gyx])
    vdp1 = lambda T, x: A.dot(np.array([x[0], x[1], x[2], x[3]])) + B
    sol = solve_ivp (vdp1, [0, 4], np.array([xi0, yi0, xp0, yp0]), max_step=0.1)
    return sol

def calculateODEMatrixVector(A, B, x): 
    vdp1 = lambda T, x: A.dot(np.array([x[0], x[1], x[2], x[3]])) + B
    sol = solve_ivp (vdp1, [0, 30], np.array([x[0], x[1], x[2], x[3]]), max_step=0.1)
    return sol

def randomize_parameter_network(G, number_of_nodes):
    for i in range(number_of_nodes):
        for j in range(i+1, number_of_nodes):
            G[i][j]['parameters'] = [random_starting_matrix(), random_starting_vector(), xy_starting_vector()]
            calculatedODE = calculateODEMatrixVector(G[i][j]['parameters'][0], G[i][j]['parameters'][1], G[i][j]['parameters'][2])
            G[i][j]["data"] = calculatedODE.y
            G[i][j]["t"] = calculatedODE.t
    return G
