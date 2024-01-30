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

def normalize_values(G, normalize1checkboxvalue):
    matrixA = [
        [[], [], [], []], 
        [[], [], [], []], 
        [[], [], [], []], 
        [[], [], [], []]
    ]
    vectorB = [[], [], [], []]
    for i, j, d in G.edges(data=True):
        a = d['parameters'][0]
        b = d['parameters'][1]
        for row in range(len(a)):
            for col in range(len(a[0])):
                matrixA[row][col].append(a[row][col])
        for row in range(len(b)):
            vectorB[row].append(b[row])

    for row in range(len(matrixA)):
            for col in range(len(matrixA[0])):
                norm = np.array(matrixA[row][col])/np.linalg.norm(matrixA[row][col])
                final_vector = norm - np.mean(norm)
                if normalize1checkboxvalue:
                    scaling_factor = np.max(np.abs(final_vector))
                    final_vector = final_vector / scaling_factor
                matrixA[row][col] = final_vector
    vectorBp = [[], [], [], []]
    for row in range(len(vectorB)):
        for i in range(len(vectorB[row])):
            vectorBp[row].append(vectorB[row][i])
    for row in range(len(vectorBp)):
        norm = np.array(vectorBp[row])/np.linalg.norm(vectorBp[row])
        final_vector = norm - np.mean(norm)
        if normalize1checkboxvalue:
            scaling_factor = np.max(np.abs(final_vector))
            final_vector = final_vector / scaling_factor
        vectorBp[row] = final_vector
    
    counter = 0
    for i, j, d in G.edges(data=True):
        matrixa = np.array([[matrixA[0][0][counter], matrixA[0][1][counter], matrixA[0][2][counter], matrixA[0][3][counter]], [matrixA[1][0][counter], matrixA[1][1][counter], matrixA[1][2][counter], matrixA[1][3][counter]], [matrixA[2][0][counter], matrixA[2][1][counter], matrixA[2][2][counter], matrixA[2][3][counter]], [matrixA[3][0][counter], matrixA[3][1][counter], matrixA[3][2][counter], matrixA[3][3][counter]]])
        G[i][j]['parameters'][0] = matrixa

        vectorb = np.array([vectorBp[0][counter], vectorBp[1][counter], vectorBp[2][counter], vectorBp[3][counter]])
        G[i][j]['parameters'][1] = vectorb
        counter += 1
    return G