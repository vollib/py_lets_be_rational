from __future__ import division
import unittest

import lets_be_rational

from math import log
from math import sqrt


class TestLetsBeRationalPackage(unittest.TestCase):
    """Test the new lets_be_rational package name"""

    def _assertAlmostEqual(self, actual, expected, epsilon=1.0e-12):
        if actual is None or expected is None:
            self.fail("{} != {}".format(actual, expected))
        self.assertTrue(abs(actual - expected) < epsilon, "{} != {}".format(actual, expected))

    def test_black(self):
        F = 100
        K = 100
        sigma = .2
        T = .5
        q = 1  # CALL = 1 PUT = -1

        actual = lets_be_rational.black(F, K, sigma, T, q)
        expected = 5.637197779701664
        self._assertAlmostEqual(actual, expected)

    def test_implied_volatility_from_a_transformed_rational_guess(self):
        F = 100
        K = 100
        sigma = .2
        T = .5
        q = 1  # CALL = 1 PUT = -1

        price = 5.637197779701664
        actual = lets_be_rational.implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q)
        expected = 0.2
        self._assertAlmostEqual(actual, expected)

    def test_normalised_black(self):
        F = 100
        K = 95
        T = 0.5
        sigma = 0.3

        x = log(F/K)
        s = sigma * sqrt(T)

        q = -1  # CALL = 1 PUT = -1
        actual_put = lets_be_rational.normalised_black(x, s, q)
        expected_put = 0.061296663817558904
        self._assertAlmostEqual(actual_put, expected_put)

        q = 1  # CALL = 1 PUT = -1
        actual_call = lets_be_rational.normalised_black(x, s, q)
        expected_call = 0.11259558142181655
        self._assertAlmostEqual(actual_call, expected_call)

    def test_norm_cdf(self):
        z = 0.302569738839
        actual = lets_be_rational.norm_cdf(z)
        expected = 0.618891110513
        self._assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
