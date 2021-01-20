import numpy as np
import warnings
from typing import Union

import scipy.stats as st

from .os_utl import check_types, check_interval


# todo update type hints and docstrings
def moving_average(y_values, n_periods=20, exclude_zeros=False):
    """Calculate the moving average for one line (given as two lists, one
    for its x-values and one for its y-values).

    :param list|tuple|np.ndarray y_values: y-coordinate of each value.
    :param int n_periods: number of x values to use
    :return list result_y: result_y are the y-values of the line averaged for the previous n_periods.
    """

    # sanity checks
    assert isinstance(y_values, (list, tuple, np.ndarray))
    assert isinstance(n_periods, int)

    result_y, last_ys = [], []
    running_sum = 0
    # use a running sum here instead of avg(), should be slightly faster
    for y_val in y_values:
        if not (exclude_zeros & (y_val == 0)):
            last_ys.append(y_val)
            running_sum += y_val
            if len(last_ys) > n_periods:
                poped_y = last_ys.pop(0)
                running_sum -= poped_y
        result_y.append((float(running_sum) / float(len(last_ys))) if last_ys else 0)

    return result_y


@check_types(y=np.ndarray, yhat=np.ndarray)
def compute_mape(y, yhat, axis=0):
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    yhat = yhat.reshape(-1, 1) if yhat.ndim == 1 else yhat

    if any(y == 0):
        warnings.warn('True label as 0\'s. Performing WAPE (weighted absolute percentage error - '
                      'abs(y-yhat).mean()/y.mean()) instead.', stacklevel=2)

        return np.abs(y-yhat).mean(axis=axis)/y.mean(axis=axis)
    return (np.abs(y-yhat)/y).mean(axis=axis)


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


@check_types(confidence=float)
@check_interval('confidence', 0, 1)
def get_confidence_interval(y: np.ndarray, confidence: float = 0.95):
    """Compute the confidence interval for an array

    :param y: array-like object with values to compute from
    :param confidence: percentage of confidence to use
    :return: mean, lbound, ubound
    """

    y_mean = np.mean(y)
    z_score = st.norm.ppf((1 + confidence) / 2)
    ci = z_score * np.std(y) / y_mean
    return y_mean, y_mean-ci, y_mean+ci
