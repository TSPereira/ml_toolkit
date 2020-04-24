import numpy as np
import warnings

from .os_utl import check_types


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
