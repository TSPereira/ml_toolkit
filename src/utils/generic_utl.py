from typing import Iterable, Union

import numpy as np


def duplicated(l: Iterable) -> set:
    """Find which items are duplicated in a list (does not keep order)

    :param l: Iterable with items to check
    :return: set with items repeated (unique)

    Example:
        >>> a = [1, 2, 3, 1, 4, 5, 2]
        >>> duplicated(a)
        >>> {1, 2}
    """

    seen = set()
    return set(x for x in l if x in seen or seen.add(x))


def get_magnitude(x: Union[np.ndarray, Iterable, int, float]) -> Union[int, np.ndarray]:
    """Finds the magnitude of a value or array-like structure.

    :param x: value or array-like structure of values to get the magnitude of
    :return: magnitude(s) of the input value(s)
    """

    with np.errstate(divide='ignore'):
        z = np.log10(np.abs(x))

    if isinstance(z, np.ndarray):
        z[np.isneginf(z)] = 0
        return np.floor(z).astype(int)
    else:
        return int(np.floor(z)) if not np.isneginf(z) else 0


def ceil_decimal(x: Union[np.ndarray, Iterable, int, float], precision: int = 0, signed: bool = True) -> \
        Union[np.float, np.ndarray]:
    """Ceil the input to a specific precision. If signed takes the sign of the value into consideration for rounding up.
    If signed=False, applies on the absolute and applies the original sign after

    :param x: value or array-like structure of values to be ceiled
    :param precision: number of decimals to ceil to
    :param signed: Whether to consider the signal of the value before or after ceiling
    :return: ceiled values

    Example:
    >>> ceil_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6])
    >>> np.array([ 1.,  1.,  1., -0., -0., -0.])

    >>> ceil_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6], signed=False)
    >>> np.array([ 1.,  1.,  1., -1., -1., -1.])
    """
    z = np.array(x)
    return np.ceil(z * 10**precision) * 10**-precision if signed else \
        (np.ceil(np.abs(z) * 10**precision) * 10**-precision) * np.sign(z)


def floor_decimal(x: Union[np.ndarray, Iterable, int, float], precision: int = 0, signed: bool = True) -> \
        Union[np.float, np.ndarray]:
    """Floor the input to a specific precision. If signed takes the sign of the value into consideration for rounding
    up. If signed=False, applies on the absolute and applies the original sign after

    :param x: value or array-like structure of values to be floored
    :param precision: number of decimals to floor to
    :param signed: Whether to consider the signal of the value before or after flooring
    :return: floored values

    Example:
    >>> floor_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6])
    >>> np.array([ 0.,  0.,  0., -1., -1., -1.])

    >>> floor_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6], signed=False)
    >>> np.array([ 0.,  0.,  0., -0., -0., -0.])
    """
    z = np.array(x)
    return np.floor(z * 10**precision) * 10**-precision if signed else \
        (np.floor(np.abs(z) * 10**precision) * 10**-precision) * np.sign(z)
