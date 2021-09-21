import numpy as np
import warnings
from typing import Union, Optional, Tuple

import scipy.stats as st

from .os_utl import check_types, check_interval


@check_types(values=(list, tuple, np.ndarray), n_periods=int)
def moving_average(values: Union[list, tuple, np.ndarray], n_periods: int = 20, exclude_zeros: bool = False) -> list:
    """Calculate the moving average on a sequence

    Args:
        values: values to compute moving average on. Can be List, Tuple or Numpy Array.
        n_periods: Number of periods to consider for the moving average. Default: 20
        exclude_zeros: Whether to ignore 0's or not from the calculation. This might be useful when zeros represent
                       missing values and you want to ignore them. Default: False.

    Returns:
        List: values averaged for the previous n_periods.

    Examples:
        >>> moving_average(list(range(10)), 3)
        [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        >>> moving_average([3, 0, 3, 4, 3, 5, 0, 3, 4, 2, 0, 4, 3], 5)
        [3.0, 1.5, 2.0, 2.5, 2.6, 3.0, 3.0, 3.0, 3.0, 2.8, 1.8, 2.6, 2.6]

        >>> moving_average([3, 0, 3, 4, 3, 5, 0, 3, 4, 2, 0, 4, 3], 5, exclude_zeros=True)
        [3.0, 3.0, 3.0, 3.33, 3.25, 3.6, 3.6, 3.6, 3.8, 3.4, 3.4, 3.6, 3.2]
    """
    result_y, last_ys = [], []
    running_sum = 0
    # use a running sum here instead of avg(), should be slightly faster
    for y_val in values:
        if not (exclude_zeros & (y_val == 0)):
            last_ys.append(y_val)
            running_sum += y_val
            if len(last_ys) > n_periods:
                poped_y = last_ys.pop(0)
                running_sum -= poped_y
        result_y.append((float(running_sum) / float(len(last_ys))) if last_ys else 0)

    return result_y


@check_types(y=np.ndarray, yhat=np.ndarray)
def compute_mape(y: np.ndarray, yhat: np.ndarray, axis: Optional[int] = 0) -> np.ndarray:
    """Compute mean absolute percentage error. If there are 0's in the true array (y), then it will return the
    wape instead (to avoid division by 0 and consequent infinite value(s))

    Args:
        y: true values
        yhat: predicted values
        axis: axis on which to compute

    Returns:
        Numpy Array: Array containing the computed ratios

    Examples:
        >>> compute_mape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.8, 0.7, 0.6]))
        array([0.])

        >>> compute_mape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.8, 0.7, 0.6]), axis=1)
        array([0., 0., 0., 0., 0.])

        >>> compute_mape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.4, 0.7, 0.6]))
        array([0.1])

        >>> compute_mape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.4, 0.7, 0.6]), axis=1)
        array([0., 0., 0.5, 0., 0.])

        >>> compute_mape(np.array([1, 0.9, 0, 0.7, 0.6]), np.array([1, 0.9, 0.8, 0.7, 0.6]))
        UserWarning: True label as 0's. Performing WAPE (weighted absolute percentage error - abs(y-yhat).mean()/y.mean()) instead.
        return f(*args, **kwds)
        array([0.25])
    """

    y = y.reshape(-1, 1) if y.ndim == 1 else y
    yhat = yhat.reshape(-1, 1) if yhat.ndim == 1 else yhat

    if any(y == 0):
        warnings.warn('\nTrue label as 0\'s. Performing WAPE (weighted absolute percentage error - '
                      'abs(y-yhat).mean()/y.mean()) instead.', stacklevel=2)

        return compute_wape(y, yhat, axis)
    return (np.abs(y-yhat)/y).mean(axis=axis)


@check_types(y=np.ndarray, yhat=np.ndarray)
def compute_wape(y: np.ndarray, yhat: np.ndarray, axis: Optional[int] = 0) -> np.ndarray:
    """Compute weighted absolute percentage error (abs(y-yhat).mean()/y.mean()). If there are 0's in the true array
    (y), then it will return the wape instead (to avoid division by 0 and consequent infinite value(s))

    Args:
        y: true values
        yhat: predicted values
        axis: axis on which to compute

    Returns:
        Numpy Array: Array o containing the computed ratios

    Examples:
        >>> compute_wape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.8, 0.7, 0.6]))
        array([0.])

        >>> compute_wape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.8, 0.7, 0.6]), axis=1)
        array([0., 0., 0., 0., 0.])

        >>> compute_wape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.4, 0.7, 0.6]))
        array([0.1])

        >>> compute_wape(np.array([1, 0.9, 0.8, 0.7, 0.6]), np.array([1, 0.9, 0.4, 0.7, 0.6]), axis=1)
        array([0. , 0. , 0.5, 0. , 0. ])

        >>> compute_wape(np.array([1, 0.9, 0, 0.7, 0.6]), np.array([1, 0.9, 0.8, 0.7, 0.6]))
        array([0.25])
        """

    y = y.reshape(-1, 1) if y.ndim == 1 else y
    yhat = yhat.reshape(-1, 1) if yhat.ndim == 1 else yhat
    return np.abs(y - yhat).mean(axis=axis) / y.mean(axis=axis)


def var_sparse(a: np.ndarray, axis: Optional[int] = None) -> Union[np.ndarray, float]:
    """Variance of sparse matrix a

    Args:
        a: Array or matrix to calculate variance of
        axis: axis along which to calculate variance

    Returns:
        Numpy Array or Float: Variance according to mean(a^2) - mean(a)^2

    """
    a_squared = a.copy()

    if hasattr(a_squared, 'data') and (not isinstance(a_squared.data, memoryview)):
        a_squared.data **= 2
    else:
        a_squared **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


def std_sparse(a: np.ndarray, axis: int = None) -> Union[np.ndarray, float]:
    """Standard deviation of sparse matrix a

    Args:
        a: Array or matrix to calculate variance of
        axis: axis along which to calculate variance

    Returns:
        Numpy Array or Float: Standard Deviation computed as: sqrt(var(a))

    """
    return np.sqrt(var_sparse(a, axis))


@check_types(confidence=float)
@check_interval('confidence', 0, 1)
def get_mean_confidence_interval(y: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the confidence interval for the mean value of an array

    Args:
        y: array-like object with values to compute from
        confidence: ratio of confidence to use

    Returns:
        Tuple: of format (mean, lower_bound, upper_bound)

    Examples:
        >>> get_mean_confidence_interval(np.array([1,2,3,4,5]))
        (3.0, 1.7604099353908769, 4.239590064609123)

        >>> get_mean_confidence_interval(np.array([[1,2,3,4,5], [2,4,6,8,10]]))
        (array([1.5, 3. , 4.5, 6. , 7.5]),
         array([0.80704809, 1.61409618, 2.42114426, 3.22819235, 4.03524044]),
         array([ 2.19295191,  4.38590382,  6.57885574,  8.77180765, 10.96475956]))
    """
    y_mean = np.mean(y, axis=0)
    z_score = st.norm.ppf((1 + confidence) / 2)
    ci = z_score * np.std(y, axis=0) / np.sqrt(y.shape[0])
    return y_mean, y_mean-ci, y_mean+ci
