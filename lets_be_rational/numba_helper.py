try:
    from numba import jit
except ImportError:
    jit = None


def maybe_jit(*jit_args, **jit_kwargs):
    def wrapper(fun):
        if callable(jit):
            return jit(*jit_args, **jit_kwargs)(fun)
        else:
            return fun
    return wrapper
