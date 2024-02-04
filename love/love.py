import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


def update_plot(axx,axy,bxx,bxy,cxx,cxy,dxx,dxy,fxy,gxy,ayy,ayx,byy,byx,cyy,cyx,dyy,dyx,fyx,gyx,xi0,yi0,xp0,yp0):
    """Updates the plot of the ODE

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
    """
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