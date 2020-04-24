from typing import Iterable, Union

import numpy as np


def duplicated(l: Iterable) -> set:
    """Find which items are duplicated in a list (does not keep order)

    :param Iterable l: Iterable with items to check
    :return set: set with items repeated (unique)
    """

    seen = set()
    return set(x for x in l if x in seen or seen.add(x))


def var_sparse(a: np.ndarray, axis: int = None) -> Union[np.ndarray, float]:
    """Variance of sparse matrix a

    :param np.ndarray|sp.sparse.crs_matrix a: Array or matrix to calculate variance of
    :param int axis: axis along which to calculate variance

    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()

    if hasattr(a_squared, 'data') and (not isinstance(a_squared.data, memoryview)):
        a_squared.data **= 2
    else:
        a_squared **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


def std_sparse(a: np.ndarray, axis: int = None) -> Union[np.ndarray, float]:
    """ Standard deviation of sparse matrix a

    :param np.ndarray|sp.sparse.crs_matrix a: Array or matrix to calculate variance of
    :param int axis: axis along which to calculate variance

    std = sqrt(var(a))
    """
    return np.sqrt(var_sparse(a, axis))


def get_magnitude(a):
    with np.errstate(divide='ignore'):
        z = np.log10(np.abs(a))

    if isinstance(z, np.ndarray):
        z[np.isneginf(z)] = 0
        return np.floor(z).astype(int)
    else:
        return int(np.floor(z)) if not np.isneginf(z) else 0


def ceil_dec(a, precision=0, signed=True):
    return np.ceil(a * 10**precision) * 10**-precision if signed else \
        (np.ceil(np.abs(a) * 10**precision) * 10**-precision) * np.sign(a)


def floor_dec(a, precision=0, signed=True):
    return np.floor(a * 10**precision) * 10**-precision if signed else \
        (np.floor(np.abs(a) * 10**precision) * 10**-precision) * np.sign(a)
