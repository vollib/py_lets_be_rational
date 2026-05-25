import json
import math
from pathlib import Path

import pytest

import py_lets_be_rational as lbr
from py_lets_be_rational.exceptions import AboveMaximumException, BelowIntrinsicException


TEST_VALUES = Path(__file__).with_name("TestValues.json")


def _cases():
    with TEST_VALUES.open() as f:
        data = json.load(f)
    return list(zip(data["input"], data["output"]))


CASES = _cases()


def assert_close(actual, expected, abs_tol=1e-12, rel_tol=1e-10):
    tolerance = max(abs_tol, rel_tol * abs(expected))
    assert abs(actual - expected) <= tolerance, f"{actual} != {expected}"


@pytest.mark.parametrize("inputs, expected", CASES)
def test_reference_value_grid(inputs, expected):
    beta = lbr.normalised_black(inputs["x"], inputs["s"], inputs["q"])

    assert_close(lbr.norm_cdf(inputs["z"]), expected["norm_cdf"])
    assert_close(lbr.black(inputs["F"], inputs["K"], inputs["sigma"], inputs["T"], inputs["q"]), expected["black"])
    assert_close(beta, expected["normalised_black"])
    assert_close(lbr.normalised_black_call(inputs["x"], inputs["s"]), expected["normalised_black_call"])
    assert_close(lbr.normalised_vega(inputs["x"], inputs["s"]), expected["normalised_vega"])
    assert_close(
        lbr.normalised_implied_volatility_from_a_transformed_rational_guess(beta, inputs["x"], inputs["q"]),
        expected["normalised_implied_volatility_from_a_transformed_rational_guess"],
        abs_tol=1e-10,
        rel_tol=1e-8,
    )
    assert_close(
        lbr.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            beta, inputs["x"], inputs["q"], inputs["N"]
        ),
        expected["normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations"],
        abs_tol=1e-10,
        rel_tol=1e-8,
    )

    if not math.isnan(expected["implied_volatility_from_a_transformed_rational_guess"]):
        price = lbr.black(inputs["F"], inputs["K"], inputs["sigma"], inputs["T"], inputs["q"])
        assert_close(
            lbr.implied_volatility_from_a_transformed_rational_guess(
                price, inputs["F"], inputs["K"], inputs["T"], inputs["q"]
            ),
            expected["implied_volatility_from_a_transformed_rational_guess"],
            abs_tol=1e-9,
            rel_tol=1e-8,
        )
        assert_close(
            lbr.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
                price, inputs["F"], inputs["K"], inputs["T"], inputs["q"], inputs["N"]
            ),
            expected["implied_volatility_from_a_transformed_rational_guess_with_limited_iterations"],
            abs_tol=1e-9,
            rel_tol=1e-8,
        )


@pytest.mark.parametrize(
    "x,s,expected",
    [
        (10.0, 0.25, 148.4064211555775),
        (-10.0, 0.25, 0.0),
        (-3.0, 0.01, 0.0),
        (-1.0, 0.25, 1.7736606889378084e-06),
        (-0.1, 0.0001, 0.0),
    ],
)
def test_normalised_black_call_edge_regions(x, s, expected):
    assert_close(lbr.normalised_black_call(x, s), expected, abs_tol=1e-14, rel_tol=1e-10)


def test_implied_volatility_rejects_prices_outside_bounds():
    with pytest.raises(BelowIntrinsicException):
        lbr.implied_volatility_from_a_transformed_rational_guess(9.99, 110.0, 100.0, 1.0, 1)

    with pytest.raises(AboveMaximumException):
        lbr.implied_volatility_from_a_transformed_rational_guess(100.0, 110.0, 100.0, 1.0, -1)


def test_normalised_implied_volatility_rejects_below_intrinsic():
    with pytest.raises(BelowIntrinsicException):
        lbr.normalised_implied_volatility_from_a_transformed_rational_guess(0.01, 0.2, 1)


def test_compatibility_modules_reexport_dependency_functions():
    import lets_be_rational
    from cody_special.erf_cody import erfc_cody
    from piecewise_rational import rational_cubic_interpolation
    from py_lets_be_rational.erf_cody import erfc_cody as compat_erfc_cody
    from py_lets_be_rational.normaldistribution import norm_cdf
    from py_lets_be_rational.rationalcubic import rational_cubic_interpolation as compat_interpolation

    assert lets_be_rational.black is lbr.black
    assert compat_erfc_cody is erfc_cody
    assert compat_interpolation is rational_cubic_interpolation
    assert_close(norm_cdf(0.0), 0.5)
