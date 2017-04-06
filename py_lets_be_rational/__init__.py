# -*- coding: utf-8 -*-

"""
py_lets_be_rational
~~~~~~~~~~~~~~~~~~~

Pure python implementation of Peter Jaeckel's LetsBeRational.

:copyright: © 2017 Gammon Capital LLC
:license: MIT, see LICENSE for more details.

About LetsBeRational:
~~~~~~~~~~~~~~~~~~~~~

The source code of LetsBeRational resides at www.jaeckel.org/LetsBeRational.7z .

======================================================================================
Copyright © 2013-2014 Peter Jäckel.

Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.

WARRANTY DISCLAIMER
The Software is provided "as is" without warranty of any kind, either express or implied,
including without limitation any implied warranties of condition, uninterrupted use,
merchantability, fitness for a particular purpose, or non-infringement.
======================================================================================
"""

from py_lets_be_rational.lets_be_rational import black
from py_lets_be_rational.lets_be_rational import normalised_black
from py_lets_be_rational.lets_be_rational import normalised_black_call
from py_lets_be_rational.lets_be_rational import implied_volatility_from_a_transformed_rational_guess
from py_lets_be_rational.lets_be_rational import implied_volatility_from_a_transformed_rational_guess_with_limited_iterations
from py_lets_be_rational.lets_be_rational import normalised_implied_volatility_from_a_transformed_rational_guess
from py_lets_be_rational.lets_be_rational import normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations
from py_lets_be_rational.lets_be_rational import normalised_vega
from py_lets_be_rational.normaldistribution import norm_cdf
