from __future__ import division

import cProfile as profile
# import profile

import py_lets_be_rational

from math import log
from math import sqrt
import random

import time

TestCases = 10000

_F = [random.randint(10, 2000) for _ in range(TestCases)]
_K = [random.randint(10, 2000) for _ in range(TestCases)]
_sigma = [random.uniform(0, 1) for _ in range(TestCases)]
_T = [round(random.uniform(0.2, 1), 2) for _ in range(TestCases)]
_N = [random.randint(0, 3) for _ in range(TestCases)]
_z = [random.uniform(0, 1) for _ in range(TestCases)]
_x = [random.uniform(0, 1) for _ in range(TestCases)]


def run_black(py_lets_be_rational):
    start = time.clock()

    q = 1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]
        actual = py_lets_be_rational.black(F, K, sigma, T, q)
#        print(F, K, sigma, T, " CALL = ", actual)
    q = -1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]
        actual = py_lets_be_rational.black(F, K, sigma, T, q)
#        print(F, K, sigma, T, " PUT = ", actual)

    end = time.clock()
    return end - start


def run_implied_volatility_from_a_transformed_rational_guess(lets_be_rational_version):
    start = time.clock()
    binary_flag = [-1, 1]
    _q = [binary_flag[random.randint(0, 1)] for _ in range(TestCases)]  # CALL = 1 PUT = -1
    _price = [random.randint(200, 1000) for _ in range(TestCases)]
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]
        q = _q[i]
        # price = lets_be_rational_version.black(F, K, sigma, T, q)
        price = _price[i]
        try:
            actual = lets_be_rational_version.implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q)
        except:
            pass
#        print(F, K, sigma, T, " CALL implied_volatility_from_a_transformed_rational_guess = ", actual)
    end = time.clock()
    return end - start


def run_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(lets_be_rational_version):
    start = time.clock()
    price = 100.0
    q = 1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]
        N = _N[i]
        # price = lets_be_rational_version.black(F, K, sigma, T, q)
        actual = lets_be_rational_version.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            price, F, K, T, q, N)
#        print(F, K, sigma, T, N, " CALL test_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations = ", actual)
    q = -1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]
        N = _N[i]
        # price = lets_be_rational_version.black(F, K, sigma, T, q)
        actual = lets_be_rational_version.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            price, F, K, T, q, N)
#        print(F, K, sigma, T, N, " PUT test_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations = ", actual)
    end = time.clock()
    return end - start


def run_normalised_black(lets_be_rational_version):
    start = time.clock()
    q = 1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]

        x = log(F / K)
        s = sigma * sqrt(T)

        actual = lets_be_rational_version.normalised_black(x, s, q)
#        print(F, K, sigma, T, " CALL normalised_black = ", actual)
    q = -1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]

        x = log(F / K)
        s = sigma * sqrt(T)

        actual = lets_be_rational_version.normalised_black(x, s, q)
#        print(F, K, sigma, T, " PUT normalised_black = ", actual)
    end = time.clock()
    return end - start


def run_normalised_black_call(lets_be_rational_version):
    start = time.clock()
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]

        x = log(F / K)
        s = sigma * sqrt(T)

        actual = lets_be_rational_version.normalised_black_call(x, s)
#        print(F, K, sigma, T, " normalised_black_call = ", actual)
    end = time.clock()
    return end - start


def run_normalised_vega(lets_be_rational_version):
    start = time.clock()
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]

        x = log(F / K)
        s = sigma * sqrt(T)

        actual = lets_be_rational_version.normalised_vega(x, s)
#        print(F, K, sigma, T, " normalised_vega = ", actual)
    end = time.clock()
    return end - start


def run_normalised_implied_volatility_from_a_transformed_rational_guess(lets_be_rational_version):
    start = time.clock()
    q = 1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]

        x = _x[i]
        s = sigma * sqrt(T)
        beta = lets_be_rational_version.normalised_black(x, s, q)
        actual = lets_be_rational_version.normalised_implied_volatility_from_a_transformed_rational_guess(beta, x, q)
#        print(F, K, sigma, T, " CALL normalised_implied_volatility_from_a_transformed_rational_guess = ", actual)
    q = -1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]

        x = _x[i]
        s = sigma * sqrt(T)
        beta = lets_be_rational_version.normalised_black(x, s, q)
        actual = lets_be_rational_version.normalised_implied_volatility_from_a_transformed_rational_guess(beta, x, q)
#        print(F, K, sigma, T, " PUT normalised_implied_volatility_from_a_transformed_rational_guess = ", actual)
    end = time.clock()
    return end - start


def run_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(lets_be_rational_version):
    start = time.clock()
    q = 1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]
        N = _N[i]

        x = _x[i]
        s = sigma * sqrt(T)
        beta = lets_be_rational_version.normalised_black(x, s, q)
        actual = lets_be_rational_version.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            beta, x, q, N)
#        print(F, K, sigma, T, " CALL normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations = ", actual)
    q = -1  # CALL = 1 PUT = -1
    for i in range(TestCases):
        F = _F[i]
        K = _K[i]
        sigma = _sigma[i]
        T = _T[i]
        N = _N[i]

        x = _x[i]
        s = sigma * sqrt(T)
        beta = lets_be_rational_version.normalised_black(x, s, q)
        actual = lets_be_rational_version.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            beta, x, q, N)
#        print(F, K, sigma, T, " PUT normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations = ", actual)
    end = time.clock()
    return end - start


def run_norm_cdf(lets_be_rational_version):
    start = time.clock()
    for i in range(TestCases):
        z = _z[i]
        actual = lets_be_rational_version.norm_cdf(z)
#        print(z, " norm_cdf = ", actual)
    end = time.clock()
    return end - start

# profile.run("run_black(py_lets_be_rational)")
# profile.run("run_implied_volatility_from_a_transformed_rational_guess(py_lets_be_rational)")
# profile.run("run_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(py_lets_be_rational)")
# profile.run("run_normalised_black(py_lets_be_rational)")
# profile.run("run_normalised_black_call(py_lets_be_rational)")
# profile.run("run_normalised_vega(py_lets_be_rational)")
# profile.run("run_normalised_implied_volatility_from_a_transformed_rational_guess(py_lets_be_rational)")
# profile.run("run_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(py_lets_be_rational)")
# profile.run("run_norm_cdf(py_lets_be_rational)")
