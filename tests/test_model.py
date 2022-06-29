import unittest

import numpy as np
from human_randgen.model import HumanRng

class TestHumanRng(unittest.TestCase):
    def test_newton(self):
        """Check newton's root finding of the lognormal params"""
        mode = 241
        sigma = 12

        rng = HumanRng()
        u, v = rng._HumanRng__newton(mode, sigma**2)

        mode_recalc = np.exp(u-v)
        sigma_recalc = np.sqrt((np.exp(v) - 1) * np.exp(2*u + v))

        self.assertTrue(np.abs(mode_recalc - mode) < 0.1)
        self.assertTrue(np.abs(sigma_recalc - sigma) < 0.1)

    def test_fitting(self):
        """Check convergence to the ground-truth parameters after fitting"""
        rng = HumanRng()

        np_rng = np.random.default_rng(seed=0)

        mu = 5
        sigma = 2

        for _ in range(1000):
            data = np_rng.lognormal(mu, sigma, 10000)
            rng.fit(data)

        u, s = rng._HumanRng__map()
        self.assertTrue(np.abs(u - mu) < 0.1)
        self.assertTrue(np.abs(s - sigma) < 0.1)

    def test_generation(self):
        """Generation sanity check"""
        rng = HumanRng(mode=1, sigma=0.1)
        self.assertTrue(rng.rand() < 10)
        self.assertEqual(len(rng.rand(10)), 10)


if __name__ == '__main__':
    unittest.main()

