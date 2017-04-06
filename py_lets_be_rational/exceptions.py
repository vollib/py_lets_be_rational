# -*- coding: utf-8 -*-

"""
py_lets_be_rational.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

from py_lets_be_rational import constants


class VolatilityValueException(Exception):
    def __init__(self):
        Exception.__init__(self, "Volatility value out of range.")
        self.value = None

    def __init__(self, message, value):
        Exception.__init__(self, message)
        self.value = value


class BelowIntrinsicException(VolatilityValueException):
    def __init__(self):
        VolatilityValueException.__init__(self, "The volatility is below the intrinsic value.",
                                                       constants.VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC)


class AboveMaximumException(VolatilityValueException):
    def __init__(self):
        VolatilityValueException.__init__(self, "The volatility is above the maximum value.",
                                                       constants.VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM)


if __name__ == "__main__":
    try:
        raise BelowIntrinsicException
    except VolatilityValueException as e:
        if not isinstance(e, BelowIntrinsicException):
            raise Exception("Should be BelowIntrinsicException")

    try:
        raise AboveMaximumException
    except VolatilityValueException as e:
        if not isinstance(e, AboveMaximumException):
            raise Exception("Should be AboveMaximumException")
