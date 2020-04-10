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
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


def std_sparse(a: np.ndarray, axis: int = None) -> Union[np.ndarray, float]:
    """ Standard deviation of sparse matrix a

    :param np.ndarray|sp.sparse.crs_matrix a: Array or matrix to calculate variance of
    :param int axis: axis along which to calculate variance

    std = sqrt(var(a))
    """
    return np.sqrt(var_sparse(a, axis))
