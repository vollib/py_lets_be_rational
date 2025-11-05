# Test Coverage Improvement Recommendations for `lets_be_rational`

## Current Test Coverage Summary

### Currently Tested Functions (test_public_functions.py)
- ✅ `black()` - Black option pricing
- ✅ `implied_volatility_from_a_transformed_rational_guess()`
- ✅ `implied_volatility_from_a_transformed_rational_guess_with_limited_iterations()`
- ✅ `normalised_black()`
- ✅ `normalised_black_call()`
- ✅ `normalised_vega()`
- ✅ `normalised_implied_volatility_from_a_transformed_rational_guess()`
- ✅ `normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations()`
- ✅ `norm_cdf()` - Normal cumulative distribution function

**Total Coverage**: ~1867 lines of source code, ~527 lines of tests (**28% test-to-code ratio**)

---

## Recommended Test Improvements

### 1. **Missing Function Coverage** (Priority: HIGH)

#### normaldistribution.py
- ❌ `norm_pdf()` - Normal probability density function (not tested)
- ❌ `inverse_norm_cdf()` - Inverse normal CDF (not tested)

**Recommended tests:**
```python
def test_norm_pdf(self):
    """Test normal probability density function"""
    # Test at mean (should be ~0.3989)
    self.assertAlmostEqual(norm_pdf(0.0), 0.3989422804014327)

    # Test symmetry
    self.assertAlmostEqual(norm_pdf(1.0), norm_pdf(-1.0))

    # Test known values
    self.assertAlmostEqual(norm_pdf(1.96), 0.05844094, places=7)

def test_inverse_norm_cdf(self):
    """Test inverse normal CDF (quantile function)"""
    # Test median
    self.assertAlmostEqual(inverse_norm_cdf(0.5), 0.0)

    # Test round-trip property
    for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        z = inverse_norm_cdf(p)
        self.assertAlmostEqual(norm_cdf(z), p)

    # Test known quantiles
    self.assertAlmostEqual(inverse_norm_cdf(0.975), 1.96, places=2)
```

#### exceptions.py
- ❌ `BelowIntrinsicException` - Exception handling (not tested)
- ❌ `AboveMaximumException` - Exception handling (not tested)

**Recommended tests:**
```python
def test_below_intrinsic_exception(self):
    """Test that below intrinsic prices raise appropriate exception"""
    # Price below intrinsic value should raise
    with self.assertRaises(BelowIntrinsicException):
        # ATM call with price below intrinsic
        lets_be_rational.implied_volatility_from_a_transformed_rational_guess(
            price=0.001, F=100, K=90, T=0.5, q=1  # Deep ITM call, price too low
        )

def test_above_maximum_exception(self):
    """Test that excessively high prices raise appropriate exception"""
    with self.assertRaises(AboveMaximumException):
        # Unreasonably high price
        lets_be_rational.implied_volatility_from_a_transformed_rational_guess(
            price=999999, F=100, K=100, T=0.5, q=1
        )
```

### 2. **Edge Case Testing** (Priority: HIGH)

Current tests use only mid-range values. Add tests for:

```python
class TestEdgeCases(unittest.TestCase):

    def test_black_at_the_money(self):
        """Test ATM options"""
        F = K = 100
        sigma = 0.2
        T = 0.5

        call = lets_be_rational.black(F, K, sigma, T, 1)
        put = lets_be_rational.black(F, K, sigma, T, -1)

        # ATM call and put should have same price (put-call parity)
        self.assertAlmostEqual(call, put)

    def test_black_deep_in_the_money(self):
        """Test deep ITM options"""
        F, K = 150, 100  # 50% ITM
        sigma, T = 0.2, 0.5

        call = lets_be_rational.black(F, K, sigma, T, 1)

        # Deep ITM call should be worth at least intrinsic value
        intrinsic = F - K
        self.assertGreater(call, intrinsic)

    def test_black_deep_out_of_the_money(self):
        """Test deep OTM options"""
        F, K = 50, 100  # 50% OTM
        sigma, T = 0.2, 0.5

        call = lets_be_rational.black(F, K, sigma, T, 1)

        # Deep OTM call should be very small
        self.assertLess(call, 1.0)

    def test_black_zero_time(self):
        """Test at expiration (T=0)"""
        F, K = 110, 100
        sigma = 0.2
        T = 1e-10  # Nearly zero

        call = lets_be_rational.black(F, K, sigma, T, 1)

        # At expiration, should equal intrinsic value
        self.assertAlmostEqual(call, max(F - K, 0), delta=0.01)

    def test_black_very_high_volatility(self):
        """Test with very high volatility (>100%)"""
        F = K = 100
        sigma = 2.0  # 200% volatility
        T = 0.5

        call = lets_be_rational.black(F, K, sigma, T, 1)

        # Should still produce valid result
        self.assertGreater(call, 0)
        self.assertLess(call, F)  # Can't be worth more than forward

    def test_black_very_low_volatility(self):
        """Test with very low volatility"""
        F = K = 100
        sigma = 0.01  # 1% volatility
        T = 0.5

        call = lets_be_rational.black(F, K, sigma, T, 1)

        # Low vol ATM option should have small but positive value
        self.assertGreater(call, 0)
        self.assertLess(call, 1)
```

### 3. **Put-Call Parity Tests** (Priority: MEDIUM)

```python
def test_put_call_parity(self):
    """Verify put-call parity holds: C - P = F - K (for zero rates)"""
    test_cases = [
        (100, 100, 0.2, 0.5),   # ATM
        (110, 100, 0.3, 1.0),   # ITM
        (90, 100, 0.25, 0.25),  # OTM
    ]

    for F, K, sigma, T in test_cases:
        call = lets_be_rational.black(F, K, sigma, T, 1)
        put = lets_be_rational.black(F, K, sigma, T, -1)

        # C - P should equal F - K (discounted)
        self.assertAlmostEqual(call - put, F - K, delta=1e-10)
```

### 4. **Implied Volatility Round-Trip Tests** (Priority: HIGH)

```python
def test_implied_volatility_round_trip(self):
    """Test that IV calculation is inverse of pricing"""
    test_cases = [
        (100, 100, 0.15, 0.5, 1),   # ATM call
        (100, 100, 0.15, 0.5, -1),  # ATM put
        (110, 100, 0.25, 1.0, 1),   # ITM call
        (90, 105, 0.30, 0.25, -1),  # ITM put
    ]

    for F, K, sigma_input, T, q in test_cases:
        # Calculate price
        price = lets_be_rational.black(F, K, sigma_input, T, q)

        # Recover implied volatility
        sigma_output = lets_be_rational.implied_volatility_from_a_transformed_rational_guess(
            price, F, K, T, q
        )

        # Should recover original volatility
        self.assertAlmostEqual(sigma_input, sigma_output, places=10)
```

### 5. **Numerical Stability Tests** (Priority: MEDIUM)

```python
def test_normalised_black_extreme_x(self):
    """Test normalized Black with extreme moneyness"""
    # Very far OTM
    x = -5.0  # ln(F/K)
    s = 0.2
    result = lets_be_rational.normalised_black(x, s, 1)
    self.assertGreaterEqual(result, 0)

    # Very far ITM
    x = 5.0
    result = lets_be_rational.normalised_black(x, s, 1)
    self.assertGreaterEqual(result, 0)

def test_normalised_vega_small_s(self):
    """Test vega calculation with very small s"""
    x = 0.0
    s = 1e-8  # Very small
    result = lets_be_rational.normalised_vega(x, s)
    # Should still be positive and reasonable
    self.assertGreater(result, 0)
```

### 6. **Parametric/Property-Based Tests** (Priority: LOW)

```python
def test_black_monotonicity_in_vol(self):
    """Option prices should increase with volatility"""
    F = K = 100
    T = 0.5

    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if sigma > 0.1:
            price_low = lets_be_rational.black(F, K, sigma - 0.1, T, 1)
            price_high = lets_be_rational.black(F, K, sigma, T, 1)
            self.assertGreater(price_high, price_low)

def test_vega_is_positive(self):
    """Vega should always be positive"""
    test_cases = [(0.0, 0.2), (0.5, 0.3), (-0.5, 0.15)]

    for x, s in test_cases:
        vega = lets_be_rational.normalised_vega(x, s)
        self.assertGreater(vega, 0)
```

### 7. **Performance Regression Tests** (Priority: LOW)

```python
def test_performance_benchmark(self):
    """Ensure IV calculation completes in reasonable time"""
    import time

    iterations = 1000
    start = time.time()

    for _ in range(iterations):
        lets_be_rational.implied_volatility_from_a_transformed_rational_guess(
            price=5.637, F=100, K=100, T=0.5, q=1
        )

    elapsed = time.time() - start
    avg_time = elapsed / iterations

    # Should average less than 1ms per calculation
    self.assertLess(avg_time, 0.001)
```

---

## Summary of Recommendations

| Priority | Category | Tests to Add | Estimated Effort |
|----------|----------|--------------|------------------|
| HIGH | Missing functions | 6 tests | 2 hours |
| HIGH | Edge cases | 7 tests | 3 hours |
| HIGH | Round-trip IV | 1 comprehensive test | 1 hour |
| MEDIUM | Put-call parity | 1 test | 30 min |
| MEDIUM | Numerical stability | 4 tests | 2 hours |
| LOW | Property-based | 2 tests | 1 hour |
| LOW | Performance | 1 test | 30 min |

**Total: ~21 new test cases, ~10 hours of work**

**Expected improvement: Test-to-code ratio from 28% to ~60%**

---

## Implementation Plan

### Phase 1 (Week 1): Critical Coverage
1. Add tests for `norm_pdf()` and `inverse_norm_cdf()`
2. Add exception handling tests
3. Add round-trip IV tests

### Phase 2 (Week 2): Edge Cases
1. Add extreme value tests
2. Add put-call parity tests
3. Add numerical stability tests

### Phase 3 (Week 3): Quality & Performance
1. Add property-based tests
2. Add performance regression tests
3. Document test coverage metrics

---

## Tools for Measuring Coverage

Install coverage tools:
```bash
pip install coverage pytest-cov
```

Run coverage analysis:
```bash
coverage run -m pytest tests/
coverage report -m
coverage html  # Generate HTML report
```

Target: **80%+ code coverage** for production-ready quality.
