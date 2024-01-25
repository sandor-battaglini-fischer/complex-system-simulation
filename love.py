import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def calculateODE(axx,axy,bxx,bxy,cxx,cxy,dxx,dxy,fxy,gxy,ayy,ayx,byy,byx,cyy,cyx,dyy,dyx,fyx,gyx,xi0,yi0,xp0,yp0): 
    A = np.array([[axx, axy, bxx, bxy], [ayx, ayy, byx, byy], [cxx, cxy, dxx, dxy], [cyx, cyy, dyx, dyy]])
    B = np.array([fxy, fyx, gxy, gyx])
    vdp1 = lambda T, x: A.dot(np.array([x[0], x[1], x[2], x[3]])) + B
    sol = solve_ivp (vdp1, [0, 4], np.array([xi0, yi0, xp0, yp0]), max_step=0.1)
    return sol


def update_plot(axx,axy,bxx,bxy,cxx,cxy,dxx,dxy,fxy,gxy,ayy,ayx,byy,byx,cyy,cyx,dyy,dyx,fyx,gyx,xi0,yi0,xp0,yp0):
    sol = calculateODE(axx,axy,bxx,bxy,cxx,cxy,dxx,dxy,fxy,gxy,ayy,ayx,byy,byx,cyy,cyx,dyy,dyx,fyx,gyx,xi0,yi0,xp0,yp0)
    [t, xa] = [sol.t, sol.y]

    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('4 Subplots', fontsize=16)

    # first subplot
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(xa[0], xa[1], s=100, cmap='viridis', alpha=0.5, c=t)
    ax.scatter(xa[2], xa[3], s=100, cmap='viridis', alpha=0.5, c=t)
    ax.plot(xa[0],xa[1], c="black", label="intimacy", alpha=0.5)
    ax.plot(xa[2],xa[3], c="red", label="passion", alpha=0.5)
    ax.set_xlabel('Xena', fontsize=16)
    ax.set_ylabel('Yorgo', fontsize=16)
    ax.grid(True)
    ax.legend()

    # second subplot
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(xa[0], xa[2], s=100, cmap='viridis', alpha=0.5, c=t)
    ax.scatter(xa[1], xa[3], s=100, cmap='viridis', alpha=0.5, c=t)
    ax.plot(xa[0],xa[2], c="black", label="Yorgo", alpha=0.5)
    ax.plot(xa[1],xa[3], c="red", label="Xena", alpha=0.5)
    ax.set_xlabel('intimacy', fontsize=16)
    ax.set_ylabel('passion', fontsize=16)

    ax.grid(True)
    ax.legend()

    # third subplot
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(xa[0], xa[1], xa[2], c=t, s=100, cmap='viridis', alpha=0.5)
    ax.set_xlabel('intimacy of Xena', fontsize=16)
    ax.set_ylabel('intimacy of Yorgo', fontsize=16)
    ax.set_zlabel('passion of Xena', fontsize=16)
    ax.grid(True)

    # fourth subplot
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(xa[0], xa[1], xa[3], c=t, cmap='viridis', alpha=0.5, s=100)
    ax.set_xlabel('intimacy of Xena', fontsize=16)
    ax.set_ylabel('intimacy of Yorgo', fontsize=16)
    ax.set_zlabel('passion of Yorgo', fontsize=16)
    ax.grid(True)

    plt.show()