import unittest
import matplotlib.pyplot as plt
import os
import numpy as np
from love.base_model import external_stress

class Tests(unittest.TestCase):
#    def test_update_plot(self):
#        """test that the update_plot function works as expected
#        """

 #       update_plot()

  #      self.assertTrue(os.path.exists('plots/love_3d.png'))
   #     self.assertTrue(os.path.exists('plots/love_xy.png'))
 #       self.assertTrue(os.path.exists('plots/love_xz.png'))
 #       self.assertTrue(os.path.exists('plots/love_yz.png'))

 #       self.assertTrue(plt.fignum_exists(1))
 #       self.assertTrue(plt.fignum_exists(2))
 #       self.assertTrue(plt.fignum_exists(3))
 #       self.assertTrue(plt.fignum_exists(4))
        
    def test_love_dynamics_output_type(self):
        y = [0.895, 1.5]
        t = 0
        p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        epsilon = 0.1
        omega = 2 * np.pi / 52
        result = love_dynamics(y, t, p, epsilon, omega)
        self.assertIsInstance(result, list)  # or np.ndarray, depending on implementation

    def test_compute_LLE_for_params_output_range(self):
        param_tuple = (0.1, 0.2)
        lle_value = compute_LLE_for_params(param_tuple)
        self.assertIsInstance(lle_value, float)
        self.assertTrue(0 <= lle_value <= 10)


if __name__ == '__main__':
    unittest.main()
