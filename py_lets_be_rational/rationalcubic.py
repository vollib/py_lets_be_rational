# -*- coding: utf-8 -*-

"""
py_lets_be_rational.rationalcubic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

# Based on
#
# “Shape preserving piecewise rational interpolation”, R. Delbourgo, J.A. Gregory - SIAM journal on scientific and
# statistical computing, 1985 - SIAM. http://dspace.brunel.ac.uk/bitstream/2438/2200/1/TR_10_83.pdf  [caveat emptor:
# there are some typographical errors in that draft version]
#
from __future__ import division
from math import fabs, sqrt

from py_lets_be_rational.numba_helper import maybe_jit
from py_lets_be_rational.constants import DBL_EPSILON
from py_lets_be_rational.constants import DBL_MIN
from py_lets_be_rational.constants import DBL_MAX


minimum_rational_cubic_control_parameter_value = -(1 - sqrt(DBL_EPSILON))
maximum_rational_cubic_control_parameter_value = 2 / (DBL_EPSILON * DBL_EPSILON)


@maybe_jit(cache=True, nopython=True, nogil=True)
def _is_zero(x):
    return fabs(x) < DBL_MIN


@maybe_jit(cache=True)
def rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
        x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_l):
    """

    :param x_l:
    :type x_l: float
    :param x_r:
    :type x_r: float
    :param y_l:
    :type y_l: float
    :param y_r:
    :type y_r: float
    :param d_l:
    :type d_l: float
    :param d_r:
    :type d_r: float
    :param second_derivative_l:
    :type second_derivative_l: float

    :return:
    :rtype: float
    """
    h = (x_r - x_l)
    numerator = 0.5 * h * second_derivative_l + (d_r - d_l)
    if _is_zero(numerator):
        return 0
    denominator = (y_r - y_l) / h - d_l
    if _is_zero(denominator):
        return maximum_rational_cubic_control_parameter_value if numerator > 0 else minimum_rational_cubic_control_parameter_value
    return numerator / denominator


@maybe_jit(cache=True)
def minimum_rational_cubic_control_parameter(d_l, d_r, s, preferShapePreservationOverSmoothness):
    """

    :param d_l:
    :type d_l: float
    :param d_r:
    :type d_r: float
    :param s:
    :type s: float
    :param preferShapePreservationOverSmoothness:
    :type preferShapePreservationOverSmoothness: bool

    :return:
    :rtype: float
    """
    monotonic = d_l * s >= 0 and d_r * s >= 0
    convex = d_l <= s <= d_r
    concave = d_l >= s >= d_r
    if not monotonic and not convex and not concave:  # If 3==r_non_shape_preserving_target, this means revert to standard cubic.
        return minimum_rational_cubic_control_parameter_value
    d_r_m_d_l = d_r - d_l
    d_r_m_s = d_r - s
    s_m_d_l = s - d_l
    r1 = -DBL_MAX
    r2 = r1
    # If monotonicity on this interval is possible, set r1 to satisfy the monotonicity condition (3.8).
    if monotonic:
        if not _is_zero(s):  # (3.8), avoiding division by zero.
            r1 = (d_r + d_l) / s  # (3.8)
        elif preferShapePreservationOverSmoothness:  # If division by zero would occur, and shape preservation is preferred, set value to enforce linear interpolation.
            r1 = maximum_rational_cubic_control_parameter_value  # This value enforces linear interpolation.

    if convex or concave:
        if not (_is_zero(s_m_d_l) or _is_zero(d_r_m_s)):  # (3.18), avoiding division by zero.
            r2 = max(fabs(d_r_m_d_l / d_r_m_s), fabs(d_r_m_d_l / s_m_d_l))
        elif preferShapePreservationOverSmoothness:
            r2 = maximum_rational_cubic_control_parameter_value  # This value enforces linear interpolation.
    elif monotonic and preferShapePreservationOverSmoothness:
        r2 = maximum_rational_cubic_control_parameter_value  # This enforces linear interpolation along segments that are inconsistent with the slopes on the boundaries, e.g., a perfectly horizontal segment that has negative slopes on either edge.
    return max(minimum_rational_cubic_control_parameter_value, max(r1, r2))


@maybe_jit(cache=True)
def rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
        x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_r):
    """

    :param x_l:
    :type x_l: float
    :param x_r:
    :type x_r: float
    :param y_l:
    :type y_l: float
    :param y_r:
    :type y_r: float
    :param d_l:
    :type d_l: float
    :param d_r:
    :type d_r: float
    :param second_derivative_r:
    :type second_derivative_r: float

    :return:
    :rtype: float
    """
    h = (x_r - x_l)
    numerator = 0.5 * h * second_derivative_r + (d_r - d_l)
    if _is_zero(numerator):
        return 0
    denominator = d_r - (y_r - y_l) / h
    if _is_zero(denominator):
        return maximum_rational_cubic_control_parameter_value if numerator > 0 else minimum_rational_cubic_control_parameter_value
    return numerator / denominator


@maybe_jit(cache=True)
def convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
        x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_r,
        preferShapePreservationOverSmoothness):
    """

    :param x_l:
    :type x_l: float
    :param x_r:
    :type x_r: float
    :param y_l:
    :type y_l: float
    :param y_r:
    :type y_r: float
    :param d_l:
    :type d_l: float
    :param d_r:
    :type d_r: float
    :param second_derivative_r:
    :type second_derivative_r: float
    :param preferShapePreservationOverSmoothness:
    :type preferShapePreservationOverSmoothness: bool

    :return:
    :rtype: float
    """
    r = rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
        x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_r)
    r_min = minimum_rational_cubic_control_parameter(
        d_l, d_r, (y_r - y_l) / (x_r - x_l), preferShapePreservationOverSmoothness)
    return max(r, r_min)


@maybe_jit(cache=True)
def rational_cubic_interpolation(x, x_l, x_r, y_l, y_r, d_l, d_r, r):

    """

    :param x:
    :type x: float
    :param x_l:
    :type x_l: float
    :param x_r:
    :type x_r: float
    :param y_l:
    :type y_l: float
    :param y_r:
    :type y_r: float
    :param d_l:
    :type d_l: float
    :param d_r:
    :type d_r: float
    :param r:
    :type r: float

    :return:
    :rtype: float
    """
    h = (x_r - x_l)
    if fabs(h) <= 0:
        return 0.5 * (y_l + y_r)
    # r should be greater than -1. We do not use  assert(r > -1)  here in order to allow values such as NaN to be propagated as they should.
    t = (x - x_l) / h
    if not (r >= maximum_rational_cubic_control_parameter_value):
        t = (x - x_l) / h
        omt = 1 - t
        t2 = t * t
        omt2 = omt * omt
        # Formula (2.4) divided by formula (2.5)
        return (y_r * t2 * t + (r * y_r - h * d_r) * t2 * omt + (r * y_l + h * d_l) * t * omt2 + y_l * omt2 * omt) / (1 + (r - 3) * t * omt)

    # Linear interpolation without over-or underflow.
    return y_r * t + y_l * (1 - t)


@maybe_jit(cache=True)
def convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
        x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_l, preferShapePreservationOverSmoothness):
    """

    :param x_l:
    :type x_l: float
    :param x_r:
    :type x_r: float
    :param y_l:
    :type y_l: float
    :param y_r:
    :type y_r: float
    :param d_l:
    :type d_l: float
    :param d_r:
    :type d_r: float
    :param second_derivative_l:
    :type second_derivative_l: float
    :param preferShapePreservationOverSmoothness:
    :type preferShapePreservationOverSmoothness: bool

    :return:
    :rtype float
    """
    r = rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
        x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_l)
    r_min = minimum_rational_cubic_control_parameter(
        d_l, d_r, (y_r - y_l) / (x_r - x_l), preferShapePreservationOverSmoothness)
    return max(r, r_min)
