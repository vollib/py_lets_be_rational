from __future__ import division
import unittest

import py_lets_be_rational

from math import log
from math import sqrt


class TestPublicFunctions(unittest.TestCase):

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

        actual = py_lets_be_rational.black(F, K, sigma, T, q)
        expected = 5.637197779701664
        self._assertAlmostEqual(actual, expected)

    def test_implied_volatility_from_a_transformed_rational_guess(self):
        F = 100
        K = 100
        sigma = .2
        T = .5
        q = 1  # CALL = 1 PUT = -1

        price = 5.637197779701664
        actual = py_lets_be_rational.implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q)
        expected = 0.2
        self._assertAlmostEqual(actual, expected)

    def test_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(self):
        F = 100
        K = 100
        sigma = .232323232
        T = .5
        q = 1  # CALL = 1 PUT = -1
        N = 1

        price = 6.54635543387
        actual = py_lets_be_rational.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price, F, K, T, q, N)
        expected = 0.232323232
        self._assertAlmostEqual(actual, expected)

    def test_normalised_black(self):
        F = 100
        K = 95
        T = 0.5
        sigma = 0.3

        x = log(F/K)
        s = sigma * sqrt(T)

        q = -1  # CALL = 1 PUT = -1
        actual_put = py_lets_be_rational.normalised_black(x, s, q)
        expected_put = 0.061296663817558904
        self._assertAlmostEqual(actual_put, expected_put)

        q = 1  # CALL = 1 PUT = -1
        actual_call = py_lets_be_rational.normalised_black(x, s, q)
        expected_call = 0.11259558142181655
        self._assertAlmostEqual(actual_call, expected_call)

    def test_normalised_black_call(self):
        F = 100
        K = 95
        T = 0.5
        sigma = 0.3

        x = log(F/K)
        s = sigma * sqrt(T)

        actual = py_lets_be_rational.normalised_black_call(x, s)
        expected = 0.11259558142181655
        self._assertAlmostEqual(actual, expected)

    def test_normalised_vega(self):
        x = 0.0
        s = 0.0
        actual = py_lets_be_rational.normalised_vega(x, s)
        expected = 0.3989422804014327
        self._assertAlmostEqual(actual, expected)

        x = 0.0
        s = 2.937528694999807
        actual = py_lets_be_rational.normalised_vega(x, s)
        expected = 0.13566415614561067
        self._assertAlmostEqual(actual, expected)

        x = 0.0
        s = 0.2
        actual = py_lets_be_rational.normalised_vega(x, s)
        expected = 0.3969525474770118
        self._assertAlmostEqual(actual, expected)

    def test_normalised_implied_volatility_from_a_transformed_rational_guess(self):
        x = 0.0
        s = 0.2
        q = 1  # CALL = 1 PUT = -1
        beta_call = py_lets_be_rational.normalised_black(x, s, q)
        actual = py_lets_be_rational.normalised_implied_volatility_from_a_transformed_rational_guess(beta_call, x, q)
        expected = 0.2
        self._assertAlmostEqual(actual, expected)

        x = 0.1
        s = 0.23232323888
        q = -1  # CALL = 1 PUT = -1
        beta_put = py_lets_be_rational.normalised_black(x, s, q)
        actual = py_lets_be_rational.normalised_implied_volatility_from_a_transformed_rational_guess(beta_put, x, q)
        expected = 0.23232323888
        self._assertAlmostEqual(actual, expected)

    def test_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(self):
        x = 0.0
        s = 0.2
        q = 1  # CALL = 1 PUT = -1
        N = 1
        beta_call = py_lets_be_rational.normalised_black(x, s, q)
        actual = py_lets_be_rational.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta_call, x, q, N)
        expected = 0.2
        self._assertAlmostEqual(actual, expected)

        x = 0.1
        s = 0.23232323888
        q = -1  # CALL = 1 PUT = -1
        N = 1
        beta_put = py_lets_be_rational.normalised_black(x, s, q)
        actual = py_lets_be_rational.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta_put, x, q, N)
        expected = 0.23232323888
        self._assertAlmostEqual(actual, expected)

    def test_norm_cdf(self):
        z = 0.302569738839
        actual = py_lets_be_rational.norm_cdf(z)
        expected = 0.618891110513
        self._assertAlmostEqual(actual, expected)

        z = 0.161148382602
        actual = py_lets_be_rational.norm_cdf(z)
        expected = 0.564011732814
        self._assertAlmostEqual(actual, expected)

