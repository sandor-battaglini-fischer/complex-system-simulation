import unittest
import matplotlib.pyplot as plt
import os
import numpy as np
from Sandor_6D_dynamical_system import update_plot
from Sandor_chaotic_love_base_model import love_dynamics
from Sandor_chaotic_love_varying_appeal import enviromental_stress_plot
from Sandor_external_stress_lyapunov_simulation import largest_lyapunov_exponent

class Tests(unittest.TestCase):
    def test_update_plot(self):

        update_plot()

        self.assertTrue(os.path.exists('plots/love_3d.png'))
        self.assertTrue(os.path.exists('plots/love_xy.png'))
        self.assertTrue(os.path.exists('plots/love_xz.png'))
        self.assertTrue(os.path.exists('plots/love_yz.png'))

        self.assertTrue(plt.fignum_exists(1))
        self.assertTrue(plt.fignum_exists(2))
        self.assertTrue(plt.fignum_exists(3))
        self.assertTrue(plt.fignum_exists(4))
        
    def test_love_dynamics(self):
        y = [1, 2]
        t = 0
        p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

        result = love_dynamics(y, t, p)

        expected_result = [0.49984814373167274, 0.5240979038175116]

        np.testing.assert_allclose(result, expected_result, rtol=1e-5)
        
    def test_enviromental_stress_plot(self):
        # Sample data
        t = np.linspace(0, 10, 100)
        solution = np.random.rand(100, 2)

        enviromental_stress_plot(t, solution)

        # Check if the plot is generated
        assert plt.gcf().get_axes()[0].has_data()

        # Check if the plot has the correct labels
        assert plt.gcf().get_axes()[0].get_xlabel() == 'Time (weeks)'
        assert plt.gcf().get_axes()[0].get_ylabel() == 'Feelings'
        assert plt.gcf().get_axes()[0].get_title() == 'Dynamics of Romantic Relationship with Environmental Stress'

        # Check if the legend is displayed
        assert plt.gcf().get_axes()[0].get_legend() is not None

        # Check if the second plot is generated
        assert plt.gcf().get_axes()[1].has_data()

        # Check if the second plot has the correct labels
        assert plt.gcf().get_axes()[1].get_xlabel() == r'$x_{1, h}$ (Partner 1 Peak h)'
        assert plt.gcf().get_axes()[1].get_ylabel() == r'$x_{1, h+1}$ (Partner 1 Peak h+1)'
        assert plt.gcf().get_axes()[1].get_title() == 'Peak-to-Peak Plot (PPP) for Partner 1'

        plt.close()
        
    def test_lyapunov_plots(self):
        # Test Case 1: Basic test case with default parameters
        initial_conditions = np.array([1, 2])
        A1 = 0.5
        epsilon = 0.1
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        omega = 0.01
        expected_output = 0.0
        output = largest_lyapunov_exponent(initial_conditions, A1, epsilon, params, omega)
        assert np.isclose(output, expected_output), f"Test Case 1 failed: Expected {expected_output}, but got {output}"

        # Test Case 2: Test with non-zero initial conditions
        initial_conditions = np.array([1, 2])
        A1 = 0.5
        epsilon = 0.1
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        omega = 0.01
        expected_output = 0.0
        output = largest_lyapunov_exponent(initial_conditions, A1, epsilon, params, omega)
        assert np.isclose(output, expected_output), f"Test Case 2 failed: Expected {expected_output}, but got {output}"

        # Test Case 3: Test with different parameters
        initial_conditions = np.array([1, 2])
        A1 = 1.0
        epsilon = 0.5
        params = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        omega = 0.05
        expected_output = 0.0
        output = largest_lyapunov_exponent(initial_conditions, A1, epsilon, params, omega)
        assert np.isclose(output, expected_output), f"Test Case 3 failed: Expected {expected_output}, but got {output}"

        # Test Case 4: Test with larger initial conditions
        initial_conditions = np.array([10, 20])
        A1 = 0.5
        epsilon = 0.1
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        omega = 0.01
        expected_output = 0.0
        output = largest_lyapunov_exponent(initial_conditions, A1, epsilon, params, omega)
        assert np.isclose(output, expected_output), f"Test Case 4 failed: Expected {expected_output}, but got {output}"


if __name__ == '__main__':
    unittest.main()