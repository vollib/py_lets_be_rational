import os


try:
    from numba import jit
except ImportError:
    jit = None


def _numba_enabled():
    return os.environ.get("PY_LETS_BE_RATIONAL_ENABLE_NUMBA", "").lower() in {"1", "true", "yes", "on"}


def maybe_jit(*jit_args, **jit_kwargs):
    def wrapper(fun):
        if callable(jit) and _numba_enabled():
            return jit(*jit_args, **jit_kwargs)(fun)
        else:
            return fun
    return wrapper
