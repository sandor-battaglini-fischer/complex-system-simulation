import networkx as nx
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
import plotly.graph_objects as go
import pandas as pd
import multiprocessing
import tqdm
num = 0
random.seed(0)


def calculateODE(initial_conditions, params):
    """Calculates the ODE for the triangular relationship

    Returns:
        array: array of solve_ivp solutions
    """
    A = np.array([[axx, axy, bxx, bxy, lxx, lxy],
                  [ayx, ayy, byx, byy, lyx, lyy],
                  [cxx, cxy, dxx, dxy, nxx, nxy],
                  [cyx, cyy, dyx, dyy, nyx, nyy],
                  [mxx, mxy, oxx, oxy, pxx, pxy],
                  [myx, myy, oyx, oyy, pyx, pyy]])
    B = np.array([fxy, fyx, gxy, gyx, hxy, hyx])
    vdp1 = lambda T, x: A.dot(x) + B
    sol = solve_ivp(vdp1, [0, 8], np.array([xi0, yi0, xp0, yp0, xc0, yc0]), max_step=0.1)
    return sol


def largest_lyapunov_exponent(initial_conditions, params, delta=0.0001, T=208, dt=0.02):
    """calculates the largest lyapunov exponent

    Args:
        initial_conditions (list): list of initial conditions
        params (list): list of parameters
        delta (float, optional): delta value . Defaults to 0.0001.
        T (int, optional): length of time to calculate. Defaults to 208.
        dt (float, optional): delta time. Defaults to 0.02.

    Returns:
        int: largest lyapunov exponent
    """
    t = np.arange(0, T, dt)
    n = len(t)

    # Compute trajectories
    sol1 = calculateODE(initial_conditions, params)
    perturbed_initial = initial_conditions + np.random.normal(0, delta, len(initial_conditions))
    sol2 = calculateODE(perturbed_initial, params)

    # Calculate divergence
    divergence = np.linalg.norm(sol2.y - sol1.y, axis=1)

    # Avoid divide by zero in log: add a small constant or filter out zeros
    epsilon = 1e-10  # Small constant to avoid log(0)
    divergence = np.maximum(divergence, epsilon)

    # Compute Lyapunov Exponent
    lyapunov = 1/n * np.sum(np.log(divergence/delta))

    return lyapunov

def compute_LLE_for_random_params(initial_conditions, var, params):
    """Computes the largest lyapunov exponent for the given parameters

    Args:
        initial_conditions (list): list of initial conditions
        var (float): variation of the initial conditions
        params (list): list of parameters

    Returns:
        float: the value of the largest lyapunov exponent
    """
    return largest_lyapunov_exponent(initial_conditions, var, params)



# Parameters
# Intimacy and Passion Dynamics for Xena
axx = -0.38  # Forgetting coefficient of the intimacy of Xena to Yorgo.
axy = 0.4  # If Yorgo’s intimacy increases, Xena’s will decrease, and if it decreases, it will increase.
bxx = +0.8  # If Xena’s passion increases, her sense of intimacy increases, and if it decreases, it decreases.
bxy = 0.0  # As Yorgo’s passion for Xena increase, Xena’s closeness to Yorgo decreases.
cxx = +0.1  # Her passion increase when Xena feels close. Men with whom she does not feel close are not attractive.
cxy = +0.93  # Intimate men are very attractive to Xena. Her passion for men who do not behave closely is significantly reduced.
dxx = -0.05  # Forgetting coefficient of the passion of Xena for Yorgo.
dxy = +0.4  # As Yorgo’s passion grows, so does Xena’s. A man who acts romantic may attract her.

# Intimacy and Passion Dynamics for Yorgo
ayy = 0.9  # Forgetting coefficient of the intimacy of Yorgo to Xena.
ayx = -0.7  # If Yorgo’s intimacy increases, Xena’s will decrease, if it decreases, it will increase.
byy = -0.9  # If Yorgo’s passion increases, his sense of intimacy decreases, and if it decreases, it increases.
byx = -0.6  # As Xena’s passion for Yorgo increases, Yorgo’s intimacy with Xena increases.
cyy = 0.1  # His passion decreases when Yorgo feels close. Women with whom he does not feel close are attractive.
cyx = -0.1  # Intimate women are not attractive to Yorgo. His passion for women who are close to him weakens a little.
dyy = -0.3  # Forgetting coefficient of the passion of Yorgo to Xena.
dyx = -0.8  # Effect of Xena's passion on Yorgo's passion. As Xena’s passion increases, Yorgo’s decreases.

# Impression coefficients
fxy = -0.7  # Xena’s impression of intimacy or friendship with Yorgo. She finds Yorgo intimate and friendly.
gyx = -0.34  # Yorgo’s impression of glamourousness or attractiveness about Xena. He finds Xena attractive and desires her romantically and sexually.
fyx = -0.1  # Yorgo’s impression of intimacy or friendship with Xena. Yorgo found Xena neither sympathetic nor antipathetic.
gxy = -0.3  # Xena’s impression of glamourousness or attractiveness about Yorgo. She does not find Yorgo romantically or sexually attractive.
hxy = +0.0  # Xena's impression of commitment to Yorgo. She does not find Yorgo committed to her.
hyx = +0.0  # Yorgo's impression of commitment to Xena. He does not find Xena committed to him.


# Commitment Dynamics
lxx = 0.3  # Influence of Xena's commitment on her own intimacy. She's more intimate with Yorgo because she's committed to him.
lxy = 0.2  # Influence of Yorgo's commitment on Xena's intimacy. She is more intimate with Yorgo because he is committed to her.
lyy = 0.1  # Influence of Yorgo's commitment on his own intimacy. He is more intimate with Xena because he is committed to her.
lyx = 0.1  # Influence of Xena's commitment on Yorgo's intimacy. The more she is committed to him, the more intimate he is with her.

mxx = 0.1  # Influence of Xena's intimacy on her own commitment. She is committed to Yorgo because she feels close to him.
mxy = 0.5  # Influence of Yorgo's intimacy on Xena's commitment. She is committed to Yorgo because he feels close to her.
myy = 0.0  # Influence of Yorgo's intimacy on his own commitment. His intimacy does not affect his commitment.
myx = 0.1  # Influence of Xena's intimacy on Yorgo's commitment. Her intimacy makes him slightly more committed to her.

nxx = -0.3  # Influence of Xena's commitment on her own passion. The more she is committed to Yorgo, the less passionate she is about him.
nxy = 0.4  # Influence of Yorgo's commitment on Xena's passion. The more he is committed to her, the more passionate she is about him.
nyy = 0.1  # Influence of Yorgo's commitment on his own passion. The more he is committed to Xena, the more passionate he is about her.
nyx = 0.2  # Influence of Xena's commitment on Yorgo's passion. The more she is committed to him, the more passionate he is about her.

oxx = -0.1  # Influence of Xena's passion on her own commitment. Her own passion scares her away.
oxy = 0.7  # Influence of Yorgo's passion on Xena's commitment. Yorgo's passion makes her more committed to him.
oyy = 0.2  # Influence of Yorgo's passion on his own commitment. The more he is passionate about Xena, the more committed he is to her.
oyx = 0.3  # Influence of Xena's passion on Yorgo's commitment. The more she is passionate about him, the more committed he is to her.

pxx = -0.2   # Forgetting coefficient of the commitment of Xena to Yorgo.
pxy = 0.1   # As Yorgo's commitment increases, Xena's commitment increases.
pyx = -0.1   # As Xena's commitment increases, Yorgo's commitment decreases.
pyy = -0.1   # Forgetting coefficient of the commitment of Yorgo to Xena.


# Initial Conditions
xi0 = 0.0   # Initial intimacy level for Xena
yi0 = 0.0   # Initial intimacy level for Yorgo
xp0 = 0.0   # Initial passion level for Xena
yp0 = 0.0   # Initial passion level for Yorgo
xc0 = 0.0   # Initial commitment level for Xena
yc0 = 0.0   # Initial commitment level for Yorgo



initial_conditions = [xi0, yi0, xp0, yp0, xc0, yc0]
params = [axx, axy, bxx, bxy, cxx, cxy, dxx, dxy, fxy, fyx, gxy, gyx, ayx, ayy, byx, byy, cyx, cyy, dyx, dyy, hxy, hyx, lxx, lxy, lyx, lyy, mxx, mxy, myx, myy, nxx, nxy, nyx, nyy, oxx, oxy, oyx, oyy, pxx, pxy, pyx, pyy]
lle = largest_lyapunov_exponent(initial_conditions, params)


def compute_sensitivity_for_random_params(args):
    """Computes the sensitivity for the given parameters

    Args:
        args (list): list of parameters

    Returns:
        list: list of sensitivities
    """
    initial_conditions, params = args
    return sensitivity_analysis(initial_conditions, params)



def sensitivity_analysis(initial_conditions, params, num_runs=10000):
    """Performs sensitivity analysis for the given parameters

    Args:
        initial_conditions (list): list of initial conditions
        params (list): list of parameters
        num_runs (int, optional): number of runs. Defaults to 10000.

    Returns:
        list: list of sensitivities
    """
    sensitivities = {f'param_{i}': [] for i in range(len(params))}

    for _ in range(num_runs):
        base_lle = largest_lyapunov_exponent(initial_conditions, params)
        for i in range(len(params)):
            varied_params = params.copy()
            variation_factor = random.uniform(-1, 1)
            varied_params[i] += variation_factor

            varied_lle = largest_lyapunov_exponent(initial_conditions, varied_params)

            if variation_factor != 0:
                sensitivity = (varied_lle - base_lle) / variation_factor
            else:
                sensitivity = 0

            sensitivities[f'param_{i}'].append(sensitivity)

    return sensitivities


def random_parameters_sampling(num_samples):
    """Randomly samples the parameters

    Args:
        num_samples (int): number of samples

    Returns:
        list: list of sampled parameters
    """
    axx_range = (-1, 1)
    axy_range = (-1, 1)
    ayx_range = (-1, 1)
    ayy_range = (-1, 1)
    bxx_range = (-1, 1)
    bxy_range = (-1, 1)
    byx_range = (-1, 1)
    byy_range = (-1, 1)
    cxx_range = (-1, 1)
    cxy_range = (-1, 1)
    cyx_range = (-1, 1)
    cyy_range = (-1, 1)
    dxx_range = (-1, 1)
    dxy_range = (-1, 1)
    dyx_range = (-1, 1)
    dyy_range = (-1, 1)
    fxy_range = (-1, 1)
    fyx_range = (-1, 1)
    gxy_range = (-1, 1)
    gyx_range = (-1, 1)
    hxy_range = (-1, 1)
    hyx_range = (-1, 1)
    lxx_range = (-1, 1)
    lxy_range = (-1, 1)
    lyx_range = (-1, 1)
    lyy_range = (-1, 1)
    mxx_range = (-1, 1)
    mxy_range = (-1, 1)
    myx_range = (-1, 1)
    myy_range = (-1, 1)
    nxx_range = (-1, 1)
    nxy_range = (-1, 1)
    nyx_range = (-1, 1)
    nyy_range = (-1, 1)
    oxx_range = (-1, 1)
    oxy_range = (-1, 1)
    oyx_range = (-1, 1)
    oyy_range = (-1, 1)
    pxx_range = (-1, 1)
    pxy_range = (-1, 1)
    pyx_range = (-1, 1)
    pyy_range = (-1, 1)
    
    

    # Random sampling
    sampled_params = []
    for _ in range(num_samples):
        axx = random.uniform(*axx_range)
        axy = random.uniform(*axy_range)
        ayx = random.uniform(*ayx_range)
        ayy = random.uniform(*ayy_range)
        bxx = random.uniform(*bxx_range)
        bxy = random.uniform(*bxy_range)
        byx = random.uniform(*byx_range)
        byy = random.uniform(*byy_range)
        cxx = random.uniform(*cxx_range)
        cxy = random.uniform(*cxy_range)
        cyx = random.uniform(*cyx_range)
        cyy = random.uniform(*cyy_range)
        dxx = random.uniform(*dxx_range)
        dxy = random.uniform(*dxy_range)
        dyx = random.uniform(*dyx_range)
        dyy = random.uniform(*dyy_range)
        fxy = random.uniform(*fxy_range)
        fyx = random.uniform(*fyx_range)
        gxy = random.uniform(*gxy_range)
        gyx = random.uniform(*gyx_range)
        hxy = random.uniform(*hxy_range)
        hyx = random.uniform(*hyx_range)
        lxx = random.uniform(*lxx_range)
        lxy = random.uniform(*lxy_range)
        lyx = random.uniform(*lyx_range)
        lyy = random.uniform(*lyy_range)
        mxx = random.uniform(*mxx_range)
        mxy = random.uniform(*mxy_range)
        myx = random.uniform(*myx_range)
        myy = random.uniform(*myy_range)
        nxx = random.uniform(*nxx_range)
        nxy = random.uniform(*nxy_range)
        nyx = random.uniform(*nyx_range)
        nyy = random.uniform(*nyy_range)
        oxx = random.uniform(*oxx_range)
        oxy = random.uniform(*oxy_range)
        oyx = random.uniform(*oyx_range)
        oyy = random.uniform(*oyy_range)
        pxx = random.uniform(*pxx_range)
        pxy = random.uniform(*pxy_range)
        pyx = random.uniform(*pyx_range)
        pyy = random.uniform(*pyy_range)
        
  
        
        # Append the sampled parameter set
        sampled_params.append([axx, axy, ayx, ayy, bxx, bxy, byx, byy, cxx, cxy, cyx, cyy, dxx, dxy, dyx, dyy, fxy, fyx, gxy, gyx, hxy, hyx, lxx, lxy, lyx, lyy, mxx, mxy, myx, myy, nxx, nxy, nyx, nyy, oxx, oxy, oyx, oyy, pxx, pxy, pyx, pyy])

    return sampled_params

if __name__ == '__main__':
    num_samples = 10000
    initial_conditions = [0, 0, 0, 0]
    sampled_params = random_parameters_sampling(num_samples)

    start_time = time.time()

    with multiprocessing.Pool() as pool:
        args = [(initial_conditions, params) for params in sampled_params]
        results = pool.map(compute_sensitivity_for_random_params, args)

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")


    # Aggregate results
    all_sensitivities = {f'param_{i}': [] for i in range(len(sampled_params[0]))}
    for result in results:
        for key in all_sensitivities.keys():
            all_sensitivities[key].extend(result[key])

    # Analyze sensitivity of parameters
    sensitivities_df = pd.DataFrame(all_sensitivities)
    sensitivities_std = sensitivities_df.std()
    sensitive_params = sensitivities_std.sort_values(ascending=False)
    print("Parameters sensitivity ranking:")
    print(sensitive_params)

    # Save to Excel
    sensitivities_df.to_excel("sensitivities_all.xlsx", index=False)
    sensitive_params.to_excel("sensitivities_param_sensitivity.xlsx", index=True)
