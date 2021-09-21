from time import sleep
from typing import Iterable, Union, Optional, Callable

import numpy as np
import pandas as pd
from decorator import decorator


def retry(n: int, exception_types: Optional[Exception] = None, timeout: Optional[int] = None) -> Callable:
    """Decorator to retry a function multiple times before failing

    Args:
        n: number of times to retry
        exception_types: exception types to accept for retry. If any other exception is raised the function will fail
            normally
        timeout: time to wait between trials

    Returns:
        Original function
    """

    @decorator
    def try_it(func, *args, **kwargs):
        for _ in range(n-1):
            try:
                return func(*args, **kwargs)
            except exception_types or Exception:
                if timeout is not None:
                    sleep(timeout)
        else:
            return func(*args, **kwargs)

    return try_it


def duplicated(lst: Iterable) -> set:
    """Find which items are duplicated in a list/set/tuple (does not keep order)

    Args:
        lst: Iterable with items to check

    Returns:
        Set: set with items repeated (unique)

    Examples:
        >>> a = [1, 2, 3, 1, 4, 5, 2]
        >>> duplicated(a)
        {1, 2}
    """
    seen = set()
    return set(x for x in lst if x in seen or seen.add(x))


def get_magnitude(x: Union[np.ndarray, Iterable, int, float]) -> Union[int, np.ndarray]:
    """Finds the magnitude of a value or array-like structure.

    Args:
        x: value or array-like structure of values to get the magnitude of

    Returns:
        Int or Numpy Array: magnitude(s) of the input value(s)

    Examples:
        >>> get_magnitude(1000)
        3

        >>> get_magnitude(-1e6)
        6

        >>> get_magnitude(np.array([10, 1000, -100, 0]))
        np.array([1, 3, 2, 0])
    """

    with np.errstate(divide='ignore'):
        z = np.log10(np.abs(x))

    if isinstance(z, np.ndarray):
        z[np.isneginf(z)] = 0
        return np.floor(z).astype(int)
    else:
        return int(np.floor(z)) if not np.isneginf(z) else 0


def ceil_decimal(x: Union[np.ndarray, Iterable, int, float], precision: int = 0, signed: bool = True) -> \
        Union[np.float64, np.ndarray]:
    """Ceil the input to a specific precision. If signed takes the sign of the value into consideration for rounding up.
    If signed=False, applies on the absolute and applies the original sign after


    Args:
        x: value or array-like structure of values to be ceiled
        precision: number of decimals to ceil to
        signed: Whether to consider the signal of the value before or after ceiling

    Returns:
        Numpy Float or Numpy Array: ceiled values

    Examples:
        >>> ceil_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6])
        np.array([ 1.,  1.,  1., -0., -0., -0.])

        >>> ceil_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6], signed=False)
        np.array([ 1.,  1.,  1., -1., -1., -1.])
    """
    z = np.array(x)
    return np.ceil(z * 10**precision) * 10**-precision if signed else \
        (np.ceil(np.abs(z) * 10**precision) * 10**-precision) * np.sign(z)


def floor_decimal(x: Union[np.ndarray, Iterable, int, float], precision: int = 0, signed: bool = True) -> \
        Union[np.float64, np.ndarray]:
    """Floor the input to a specific precision. If signed takes the sign of the value into consideration for rounding
    up. If signed=False, applies on the absolute and applies the original sign after

    Args:
        x: value or array-like structure of values to be floored
        precision: number of decimals to floor to
        signed: Whether to consider the signal of the value before or after flooring

    Returns:
        Numpy Float or Numpy Array: floored values

    Examples:
        >>> floor_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6])
        np.array([ 0.,  0.,  0., -1., -1., -1.])

        >>> floor_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6], signed=False)
        np.array([ 0.,  0.,  0., -0., -0., -0.])
    """
    z = np.array(x)
    return np.floor(z * 10**precision) * 10**-precision if signed else \
        (np.floor(np.abs(z) * 10**precision) * 10**-precision) * np.sign(z)


def create_random_points(x0, y0, distance, n_points=1):
    """Create n_points random lat, lon coordinates within distance (in meters) of origin point
    """
    r = distance / 111320
    u, v = np.random.uniform(0, 1, n_points), np.random.uniform(0, 1, n_points)
    w = r * np.sqrt(u)
    t = 2 * np.pi * v
    x = w * np.cos(t)
    x1 = x / np.cos(y0)
    y = w * np.sin(t)
    return x0 + x1, y0 + y


def upsample_timeseries_df(df, rule='D', method='ffill', group_index=None, date_index=None):
    """
    todo move to sia_preprocessing.timeseries
    Upsample a Pandas DataFrame or Series with either a DatetimeIndex or MultiIndex.
    Workaround while this issue is not solved: https://github.com/pandas-dev/pandas/issues/28313

    Args:
        df: Pandas DataFrame or Series.
        rule: Upsampling frequency e.g. 'D' for daily.

            This is passed directly to the Pandas resampler which has more options:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        method: String for the method of filling in empty values. Valid options:
            'ffill' is forward-fill with last known values.
            'linear' is linear interpolation between known values.

        group_index: If `df` has a MultiIndex then use this argument to select indexes to groupby before resampling.
        date_index: If `df` has a MultiIndex then use this data-column as the dates.

    Returns:
        Upsampled DataFrame or Series.
    """

    assert isinstance(df, (pd.DataFrame, pd.Series))
    assert isinstance(df.index, (pd.DatetimeIndex, pd.MultiIndex))
    assert method in ('ffill', 'linear')

    # Pandas 0.25.1 does not support upsampling a DataFrame with a MultiIndex
    # using the normal resample() function, so we must handle the two cases
    # differently.
    __methods__ = dict(ffill=lambda x: x.ffill(), linear=lambda x: x.interpolate(method='linear'))

    # If the DataFrame has a DatetimeIndex.
    if isinstance(df.index, pd.DatetimeIndex):
        # Normal upsampling using Pandas.
        df_upsampled = __methods__[method](df.resample(rule))

    # If the DataFrame has a MultiIndex.
    elif isinstance(df.index, pd.MultiIndex):
        # Pandas has very complicated semantics for resampling a DataFrame
        # with a MultiIndex.

        assert date_index is not None and group_index is not None, \
            'When passing a MultiIndex you need to provide the names of the indexes to be used as group and date'

        # Helper-function for resampling a DataFrame for a single company.
        def _resample(group):
            return __methods__[method](group.set_index(date_index).resample(rule))

        # Group the original DataFrame by companies and apply the resampling to each.
        df_upsampled = df.reset_index(level=date_index).groupby(level=group_index).apply(_resample)

    return df_upsampled
