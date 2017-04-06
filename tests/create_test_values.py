import lets_be_rational
import random

TestCases = 1000

_F = [random.randint(10, 1000) for _ in range(TestCases)]
_K = [random.randint(10, 1000) for _ in range(TestCases)]
_sigma = [random.uniform(0.5, 1) for _ in range(TestCases)]
_T = [round(random.uniform(0.0, 2.0),2) for _ in range(TestCases)]
_N = [random.randint(0, 3) for _ in range(TestCases)]
_z = [random.uniform(0, 1) for _ in range(TestCases)]
_x = [random.uniform(0, 1) for _ in range(TestCases)]
_s = [random.uniform(0, 1) for _ in range(TestCases)]
_q = [-1 if random.randint(0, 1) == 0 else 1 for _ in range(TestCases)]


data = {}
data['input'] = []
data['output'] = []

for i in range(TestCases):
    F = _F[i]
    K = _K[i]
    sigma = _sigma[i]
    T = _T[i]
    q = _q[i]
    z = _z[i]
    N = _N[i]

    black = lets_be_rational.black(F, K, sigma, T, q)
    iv = lets_be_rational.implied_volatility_from_a_transformed_rational_guess(black, F, K, T, q)
    ivi = lets_be_rational.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(black, F, K, T, q, N)

    x = _x[i]
    s = _s[i]
    nblack = lets_be_rational.normalised_black(x, s, q)
    nblackcall = lets_be_rational.normalised_black_call(x, s)
    niv = lets_be_rational.normalised_implied_volatility_from_a_transformed_rational_guess(nblack, x, q)
    nivi = lets_be_rational.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(nblack, x, q, N)
    vega = lets_be_rational.normalised_vega(x, s)
    norm_cdf = lets_be_rational.norm_cdf(z)

    data['input'].append({'F':F,
                          'K':K,
                          'sigma':sigma,
                          'T':T,
                          'q':q,
                          'z':z,
                          'N':N,
                          'x':x,
                          's':s})
    data['output'].append({'black':black,
                           'implied_volatility_from_a_transformed_rational_guess':iv,
                           'implied_volatility_from_a_transformed_rational_guess_with_limited_iterations':ivi,
                           'normalised_black':nblack,
                           'normalised_black_call':nblackcall,
                           'normalised_implied_volatility_from_a_transformed_rational_guess':niv,
                           'normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations':nivi,
                           'normalised_vega':vega,
                           'norm_cdf':norm_cdf})


import json
with open('TestValues.json', 'w') as outfile:
    json.dump(data, outfile)