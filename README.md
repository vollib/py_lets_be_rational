# Overview

`lets_be_rational` (also available as `py_lets_be_rational` for backward compatibility) is a pure Python port of the original C implementation of Peter Jäckel's "Let's Be Rational" algorithm for implied volatility calculation.

## Installation

Install via pip:

```bash
pip install lets_be_rational
```

## Usage

```python
import lets_be_rational

# Calculate Black option price
price = lets_be_rational.black(F=100, K=100, sigma=0.2, T=0.5, q=1)

# Calculate implied volatility
iv = lets_be_rational.implied_volatility_from_a_transformed_rational_guess(
    price=5.637, F=100, K=100, T=0.5, q=1
)
```

**Note:** For backward compatibility, you can also use `import py_lets_be_rational`. Both package names are available.

## Comparison

Below is a list of differences between `lets_be_rational` (this Python implementation) and the original C `lets_be_rational`:

| Feature                                     | `py_lets_be_rational` | `lets_be_rational`         |
| ------------------------------------------- |:---------------------:|:--------------------------:|
| Python Version Compatibility                | 2.7 and 3.x           |           2.7 only         |
| Source Language                             | Python                | C with Python SWIG Wrapper |
| Optional Dependencies                       | Numba                 | None                       |
| Installed Automatically by `pip` as part of | py_vollib             | vollib                     |


## Execution Speed
Except for their source language, `py_lets_be_rational` and `lets_be_rational` are almost identical. Each is orders of 
magnitude faster than traditional implied volatility calculation libraries, thanks to the algorithms developed by 
Peter Jaeckel. However, `py_lets_be_rational`, without Numba installed, is about an order of magnitude slower than 
`lets_be_rational`. Numba helps to mitigate this speed gap considerably.

## Numba Dependency
Numba is an optional dependency of `py_lets_be_rational` . Because Numba installation can be tricky and OS-dependent, 
we decided to leave it up to each user to decide how and whether to install Numba. If Numba is present, execution speed 
will be faster. If not, the code will still run -- just slower.


## Installing numba

`py_lets_be_rational` optionally depends on `numba` which in turn depends on `llvm-lite`. `llvm-lite` wants LLVM 3.9 
being installed. On Mac OSX, use the latest version of HomeBrew to install `numba`'s dependencies as shown below:

```
brew install llvm@3.9
LLVM_CONFIG=/usr/local/opt/llvm@3.9/bin/llvm-config pip install llvmlite==0.16.0
pip install numba==0.31.0
```

For other operating systems, please refer to the `llvm-lite` and `numba` documentation.

## Development

Fork our repository, and make the changes on your local copy:

```
git clone git@github.com/your_username/py_lets_be_rational
cd py_lets_be_rational
pip install -e .
pip install -r dev-requirements.txt
```

When you are done, push the changes, and create a pull request on your repo.


## About "Let's be Rational"

["Let's Be Rational"](http://www.pjaeckel.webspace.virginmedia.com/LetsBeRational.pdf>) is a paper by [Peter Jäckel](http://jaeckel.org) showing *"how Black's volatility can be implied from option prices with as little as two iterations to maximum attainable precision on standard (64 bit floating point) hardware for all possible inputs."*

The paper is accompanied by the full C source code, which resides at [www.jaeckel.org/LetsBeRational.7z](www.jaeckel.org/LetsBeRational.7z).

```
    Copyright © 2013-2014 Peter Jäckel.

    Permission to use, copy, modify, and distribute this software is freely granted,
    provided that this notice is preserved.

    WARRANTY DISCLAIMER
    The Software is provided "as is" without warranty of any kind, either express or implied,
    including without limitation any implied warranties of condition, uninterrupted use,
    merchantability, fitness for a particular purpose, or non-infringement.
```
