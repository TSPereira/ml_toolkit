import numpy as np


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


def find_boundaries_hist(arr, per_samples=0.8):
    """Find the boundaries on a histogram that contain a certain percentage of samples within

    :param np.ndarray arr: array of shape (n, ) with the variable to analyse
    :param float per_samples: percentage of samples to guarantee inside of the boundaries
    :return tuple (low_bound, high_bound, per_explained): low_bound is the lowest boundary, high_bound is
    the highest boundary and per_explained is the percentage of samples within those boundaries (above the specified)
    """

    # sanity checks
    assert ((per_samples > 0) & (per_samples <= 1))
    assert len(arr.shape) == 1

    arr = arr[~np.isnan(arr)]

    # finds magnitude of number of examples to define the number of bins and calculate histogram
    _mag = np.floor(np.log10(arr.shape[0]))
    _mag = 1 if _mag <= 3 else _mag - 3
    n_bins = arr.shape[0] // int(10**_mag)
    samples, bound = np.histogram(arr, bins=n_bins)

    # iteratively add the closest bin with most examples until sum reaches per_samples
    i_min = np.argmax(samples)
    i_max = i_min + 1
    while np.sum(samples[i_min:i_max]) / arr.shape[0] < per_samples:
        if (samples[i_max] >= samples[i_min - 1]) | (i_min == 0):
            i_max += 1
        elif (samples[i_max] < samples[i_min - 1]) | (i_max == arr.shape[0] - 1):
            i_min -= 1

        if i_max > samples.shape[0] - 1:
            i_max -= 1
            i_min -= 1
        elif i_min < 0:
            i_min = 0
            i_max += 1

    per_explained = np.sum(samples[i_min:i_max])/arr.shape[0]
    low_bound = bound[i_min]
    high_bound = bound[i_max]
    return low_bound, high_bound, per_explained, n_bins


# todo join monthly and weekly stability functions before adding to closer_packages
#  add to new file "stability"
def calculate_monthly_stability(daily_sales_df_: pd.DataFrame):
    """
    Definition of new stability var as center of mass of the calendar vs sales
    :param daily_sales_df_: pandas DataFrame with the daily time series of each derivative sales
    :return: pandas Series with the stability variable for each derivative
    """

    daily_sales_df = daily_sales_df_.copy(deep=True)  # not to create the columns on the input dataframe
    if isinstance(daily_sales_df.index, pd.DatetimeIndex):
        daily_sales_df.index = daily_sales_df.index.to_series().astype(str)

    stability_df = pd.DataFrame(index=daily_sales_df.columns,
                                columns=list(set(daily_sales_df.index.str[-5:-3].astype(int))))

    daily_sales_df['day'] = daily_sales_df.index.str[-2:].astype(int)
    daily_sales_df['month'] = daily_sales_df.index.str[-5:-3].astype(int)

    def _stability(df, day_col):
        return (df.multiply(day_col, axis='index').sum()) / (df.sum())

    cols = [col for col in daily_sales_df.columns if col not in ['day', 'month']]
    total = _stability(daily_sales_df[cols], daily_sales_df['day']).fillna(0)

    for month in list(set(daily_sales_df.month)):
        aux_df = daily_sales_df[daily_sales_df.month == month]
        stability_df[month] = _stability(aux_df[cols], aux_df['day'])

    stability_df.fillna(0, inplace=True)  # replace with zeros when the division was by zero
    stability_df = stability_df.sort_index(axis=1)

    stability = np.sqrt((stability_df.sub(total, axis=0) ** 2).sum(axis=1)) / stability_df.shape[1]
    return stability_df.T, total, stability


def calculate_weekly_stability(daily_sales_df_: pd.DataFrame):
    """
    Definition of new stability var as center of mass of the calendar vs sales
    :param daily_sales_df_: pandas DataFrame with the daily time series of each derivative sales
    :return: pandas Series with the stability variable for each derivative
    """

    daily_sales_df = daily_sales_df_.copy(deep=True)  # not to create the columns on the input dataframe
    if isinstance(daily_sales_df.index, str):
        daily_sales_df.index = pd.to_datetime(daily_sales_df.index)

    weeks = daily_sales_df.index.to_series().dt.weekofyear.values

    stability_df = pd.DataFrame(index=daily_sales_df.columns,
                                columns=list(set(weeks)))

    daily_sales_df['day'] = daily_sales_df.index.to_series().dt.weekday.values
    daily_sales_df['week'] = weeks

    def _stability(df, day_col):
        return (df.multiply(day_col, axis='index').sum()) / (df.sum())

    cols = [col for col in daily_sales_df.columns if col not in ['day', 'week']]
    total = _stability(daily_sales_df[cols], daily_sales_df['day'])

    for week in list(set(daily_sales_df['week'])):
        aux_df = daily_sales_df[daily_sales_df['week'] == week]
        stability_df[week] = _stability(aux_df[cols], aux_df['day'])

    stability_df.fillna(0, inplace=True)  # replace with zeros when the division was by zero
    stability_df = stability_df.sort_index(axis=1)

    stability = np.sqrt((stability_df.sub(total, axis=0) ** 2).sum(axis=1)) / stability_df.shape[1]
    return stability_df.T, total, stability


def calculate_stabilities(skus, index_to_calc, months_to_consider=3):
    # get stability of last 3 months
    month = skus.index[-1].replace(day=1)
    end_date = month + pd.offsets.MonthEnd(1)
    start_date = month - pd.offsets.MonthBegin(months_to_consider - 1)
    skus_filt = skus.loc[(skus.index >= start_date) & (skus.index <= end_date)]
    *_, trim_stb = calculate_monthly_stability(skus_filt)

    stb_columns = ('demand', 'trimestral_stability', 'zero_clusters_stability')
    columns = pd.MultiIndex.from_tuples((skus.columns[0], col) for col in stb_columns)
    stb = pd.DataFrame(index=index_to_calc, columns=columns)

    for col in skus.columns:
        stb[(col, 'trimestral_stability')] = trim_stb[col]

        curr_period = skus.loc[index_to_calc, col]
        stb[(col, 'demand')] = curr_period.values
        stb[(col, 'zero_clusters_stability')] = perform_stability_calculation_zero_clusters(curr_period)

    return stb


def find_runs(value: int, vector: np.array):
    """
    Receives a list and a value. It identified the starting and ending indexes of all the clusters composed of
    sequences of the indicated value.
    :param value: integer to be found in the vector list.
    :param vector: list of integers on which the cluster of values will be searched for.
    :return: numpy array with inner arrays. Each one of the inner arrays contains the starting and ending indexes
    (on the original input vector) where clusters of value are found.
    """
    # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
    pad_num = 1 if value == 0 else 0
    is_value = np.pad(vector, (1, 1), 'constant', constant_values=(pad_num, pad_num)) == value
    abs_diff = np.abs(np.diff(is_value))
    return np.where(abs_diff == 1)[0].reshape(-1, 2)  # Runs start and end where abs_diff is 1.


def perform_stability_calculation_zero_clusters(array: np.array):
    """
    Calculates the numerical stability variable
    :param series: pandas Series that contains the daily sales time series of each derivative
    :return: returns an integer which is the sum of the absolute deviation of each zero cluster length to the average
    length of the zero clusters of the respective derivative
    """
    zero_sequences = find_runs(0, array)
    zero_sequences_list = zero_sequences[:, 1] - zero_sequences[:, 0]
    mean = zero_sequences_list.mean()
    return (np.abs(zero_sequences_list - mean).sum() + 1 if zero_sequences_list.shape[0] == 1 else 0) * mean