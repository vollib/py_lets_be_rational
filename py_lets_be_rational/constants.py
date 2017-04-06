# -*- coding: utf-8 -*-

"""
py_lets_be_rational.constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

from __future__ import absolute_import

from _testcapi import DBL_MIN, DBL_MAX

import sys
from math import sqrt

DBL_EPSILON = sys.float_info.epsilon

SQRT_DBL_EPSILON = sqrt(DBL_EPSILON)
FOURTH_ROOT_DBL_EPSILON = sqrt(SQRT_DBL_EPSILON)
EIGHTH_ROOT_DBL_EPSILON = sqrt(FOURTH_ROOT_DBL_EPSILON)
SIXTEENTH_ROOT_DBL_EPSILON = sqrt(EIGHTH_ROOT_DBL_EPSILON)
SQRT_DBL_MIN = sqrt(DBL_MIN)
SQRT_DBL_MAX = sqrt(DBL_MAX)

# Set this to 0 if you want positive results for (positive) denormalized inputs, else to DBL_MIN.
# Note that you cannot achieve full machine accuracy from denormalized inputs!
DENORMALIZATION_CUTOFF = 0

VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC = -DBL_MAX
VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM = DBL_MAX

ONE_OVER_SQRT_TWO = 0.7071067811865475244008443621048490392848359376887
ONE_OVER_SQRT_TWO_PI = 0.3989422804014326779399460599343818684758586311649
SQRT_TWO_PI = 2.506628274631000502415765284811045253006986740610

TWO_PI = 6.283185307179586476925286766559005768394338798750
SQRT_PI_OVER_TWO = 1.253314137315500251207882642405522626503493370305  # sqrt(pi/2) to avoid misinterpretation.
SQRT_THREE = 1.732050807568877293527446341505872366942805253810
SQRT_ONE_OVER_THREE = 0.577350269189625764509148780501957455647601751270
TWO_PI_OVER_SQRT_TWENTY_SEVEN = 1.209199576156145233729385505094770488189377498728  # 2*pi/sqrt(27)
PI_OVER_SIX = 0.523598775598298873077107230546583814032861566563
