"""Compatibility exports for piecewise rational cubic interpolation."""

from piecewise_rational import (
    convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side,
    convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side,
    minimum_rational_cubic_control_parameter,
    rational_cubic_control_parameter_to_fit_second_derivative_at_left_side,
    rational_cubic_control_parameter_to_fit_second_derivative_at_right_side,
    rational_cubic_interpolation,
)

__all__ = [
    "convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side",
    "convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side",
    "minimum_rational_cubic_control_parameter",
    "rational_cubic_control_parameter_to_fit_second_derivative_at_left_side",
    "rational_cubic_control_parameter_to_fit_second_derivative_at_right_side",
    "rational_cubic_interpolation",
]
