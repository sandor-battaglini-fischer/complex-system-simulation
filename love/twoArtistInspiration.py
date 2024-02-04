import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import ipywidgets as widgets
from ipywidgets import GridBox, Layout, interactive_output
from IPython.display import display

def love_dynamics(y, t, p):
    x1, x2 = y
    alpha1, alpha2, beta1, beta2, gamma1, gamma2, bA1, bA2, A1, A2, k1, k2, n1, n2, m1, m2, sigma1, sigma2, delta1, delta2, epsilon1, epsilon2, mu1, mu2, z1, z2 = p

    RL1 = beta1 * k1 * x2 * np.exp(-(k1 * x2)**n1)
    RL2 = beta2 * k2 * x1 * np.exp(-(k2 * x1)**n2)

    BA1 = x1**(2*m1) / (x1**(2*m1) + sigma1**(2*m1))
    BA2 = x2**(2*m2) / (x2**(2*m2) + sigma2**(2*m2))

    # Distraction factors
    z1 = epsilon1 * (mu1 * x1 - z1)
    z2 = epsilon2 * (mu2 * x2 - z2)

    dx1dt = -alpha1 * x1 + RL1 + (1 + bA1 * BA1) * gamma1 * A2 * (1/(1+ delta1 * z1))
    dx2dt = -alpha2 * x2 + RL2 + (1 + bA2 * BA2) * gamma2 * A1 * (1/(1+ delta2 * z2))

    return [dx1dt, dx2dt]

initial_conditions = [1, 1.5]
t = np.linspace(0, 50, 1000)

slider_style = {'description_width': 'initial'} 
slider_layout = Layout(width='auto')

def update_plot(alpha1, alpha2, beta1, beta2, gamma1, gamma2, bA1, bA2, A1, A2, k1, k2, n1, n2, m1, m2, sigma1, sigma2, delta1, delta2, epsilon1, epsilon2, mu1, mu2, z1, z2):
    params = [alpha1, alpha2, beta1, beta2, gamma1, gamma2, bA1, bA2, A1, A2, k1, k2, n1, n2, m1, m2, sigma1, sigma2, delta1, delta2, epsilon1, epsilon2, mu1, mu2, z1, z2]
    solution = odeint(love_dynamics, initial_conditions, t, args=(params,))
    
    plt.figure(figsize=(16, 5))

    # Time series
    plt.subplot(1, 2, 1)
    plt.plot(t, solution[:, 0], label='Romeo', color='tab:blue')
    plt.plot(t, solution[:, 1], label='Juliet', color='tab:pink')
    plt.xlabel('Time')
    plt.ylabel('Feelings')
    plt.title('Dynamics of Romantic Relationship')
    plt.legend()
    plt.grid(True)

    # Phase diagram
    plt.subplot(1, 2, 2)
    plt.plot(solution[:, 0], solution[:, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Phase Diagram')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Parameters
alpha1_slider = widgets.FloatSlider(value=0.36, min=0, max=1, step=0.01, description='Forgetting Coef. Romeo (alpha1)', style=slider_style, layout=slider_layout)
alpha2_slider = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description='Forgetting Coef. Juliet (alpha2)', style=slider_style, layout=slider_layout)
beta1_slider = widgets.FloatSlider(value=0.75, min=0, max=10, step=0.01, description='Reactiveness to Love Juliet on Romeo (beta1)', style=slider_style, layout=slider_layout)
beta2_slider = widgets.FloatSlider(value=10.66, min=0, max=20, step=0.01, description='Reactiveness to Love Romeo on Juliet (beta2)', style=slider_style, layout=slider_layout)
gamma1_slider = widgets.FloatSlider(value=1, min=0, max=5, step=0.1, description='Reactiveness to Appeal Juliet on Romeo (gamma1)', style=slider_style, layout=slider_layout)
gamma2_slider = widgets.FloatSlider(value=1, min=0, max=5, step=0.1, description='Reactiveness to Appeal Romeo on Juliet (gamma2)', style=slider_style, layout=slider_layout)
bA1_slider = widgets.FloatSlider(value=2.9, min=0, max=5, step=0.1, description='Bias Coef. of Romeo (bA1)', style=slider_style, layout=slider_layout)
bA2_slider = widgets.FloatSlider(value=1, min=0, max=5, step=0.1, description='Bias Coef. of  Juliet (bA2)', style=slider_style, layout=slider_layout)
A1_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Appeal of Romeo (A1)', style=slider_style, layout=slider_layout)
A2_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Appeal of Juliet (A2)', style=slider_style, layout=slider_layout)
k1_slider = widgets.FloatSlider(value=0.08, min=0, max=2, step=0.01, description='Insecurity of Romeo (k1)', style=slider_style, layout=slider_layout)
k2_slider = widgets.FloatSlider(value=1.5, min=0, max=2, step=0.01, description='Insecurity of Juliet (k2)', style=slider_style, layout=slider_layout)
n1_slider = widgets.FloatSlider(value=1, min=0, max=10, step=0.1, description='Shape of Reaction Function Romeo on Juliet (n1)', style=slider_style, layout=slider_layout)
n2_slider = widgets.FloatSlider(value=4, min=0, max=10, step=0.1, description='Shape of Reaction Function Juliet on Romeo (n2)', style=slider_style, layout=slider_layout)
m1_slider = widgets.FloatSlider(value=4, min=0, max=10, step=0.1, description='Shape of Bias Function Romeo (m1)', style=slider_style, layout=slider_layout)
m2_slider = widgets.FloatSlider(value=4, min=0, max=10, step=0.1, description='Shape of Bias Function Juliet (m2)', style=slider_style, layout=slider_layout)
sigma1_slider = widgets.FloatSlider(value=1, min=-1, max=1, step=0.1, description='Saddle Quantity of Romeo (sigma1)', style=slider_style, layout=slider_layout)
sigma2_slider = widgets.FloatSlider(value=1, min=-1, max=1, step=0.1, description='Saddle Quantity of Juliet (sigma2)', style=slider_style, layout=slider_layout)
delta1_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Distraction Factor Romeo (delta1)', style=slider_style, layout=slider_layout)
delta2_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Distraction Factor Juliet (delta2)', style=slider_style, layout=slider_layout)
epsilon1_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Inhibition Factor Romeo (epsilon1)', style=slider_style, layout=slider_layout)
epsilon2_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Inhibition Factor Juliet (epsilon2)', style=slider_style, layout=slider_layout)
mu1_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Love to Inspiration Factor Romeo (mu1)', style=slider_style, layout=slider_layout)
mu2_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Love to Inspiration Factor Juliet (mu2)', style=slider_style, layout=slider_layout)
z1_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Initial Inspiration Romeo (z1)', style=slider_style, layout=slider_layout)
z2_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, description='Initial Inspiration Juliet (z2)', style=slider_style, layout=slider_layout)

sliders = [
    alpha1_slider, alpha2_slider, beta1_slider, beta2_slider, gamma1_slider, gamma2_slider,
    bA1_slider, bA2_slider, A1_slider, A2_slider, k1_slider, k2_slider, n1_slider, n2_slider,
    m1_slider, m2_slider, sigma1_slider, sigma2_slider, delta1_slider, delta2_slider, 
    epsilon1_slider, epsilon2_slider, mu1_slider, mu2_slider, z1_slider, z2_slider
]

grid = GridBox(sliders, layout=Layout(
    width='100%',
    grid_template_columns='repeat(3, 32%)',  
    grid_gap='20px 20px'
))

interactive_plot = interactive_output(update_plot, {
    'alpha1': alpha1_slider, 'alpha2': alpha2_slider, 'beta1': beta1_slider, 'beta2': beta2_slider,
    'gamma1': gamma1_slider, 'gamma2': gamma2_slider, 'bA1': bA1_slider, 'bA2': bA2_slider,
    'A1': A1_slider, 'A2': A2_slider, 'k1': k1_slider, 'k2': k2_slider, 'n1': n1_slider, 
    'n2': n2_slider, 'm1': m1_slider, 'm2': m2_slider, 'sigma1': sigma1_slider, 'sigma2': sigma2_slider,
    'delta1': delta1_slider, 'delta2': delta2_slider, 'epsilon1': epsilon1_slider, 'epsilon2': epsilon2_slider,
    'mu1': mu1_slider, 'mu2': mu2_slider, 'z1': z1_slider, 'z2': z2_slider
})

widgets.VBox([grid, interactive_plot])
