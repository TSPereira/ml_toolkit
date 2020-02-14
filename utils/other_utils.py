import numpy as np
import pandas as pd
import os

from .log_utils import printv


def fill_nan_with_dist(column):
    """Finds the distribution of filled values in a column and fill its nulls with the same distribution

    :param pandas.Series column: column to fill nulls
    :return pandas.Series: column with nulls filled
    """

    # Copy to avoid direct changes in the original dataframe
    col = column.copy(deep=True)

    # Normalize the column to get the distribution in percentages
    d = col.value_counts(normalize=True)

    # Find which lines are null and fill them with a random number according to the distribution previously calculated
    col[col.isnull()] = np.random.choice(d.index, size=len(col[col.isnull()]), p=d.values)

    return col


def fill_based_other_col(data, col_to_fill, ref_col, method='mode', replace=np.nan):
    """Function to fill nulls on a DataFrame column based on other column.
    To fill based on multiple columns pass ref_col as a concatenation of the columns to use
    If replace parameter is not np.nan it will fill all columns with the passed value (for example replace=0 to replace
    non-meaningful zeros)

    :param data: DataFrame with the two columns to use/fill
    :param col_to_fill: column to fill
    :param ref_col: column to use as reference
    :param method: method to use to find the value to fill with
    :param replace: optional parameter to control the value to replace. Standard is np.nan to fill nulls
    :return: Column filled
    """

    assert method in ['mode', 'mean', 'median']
    assert {col_to_fill, ref_col}.issubset(data.columns)
    df = data.copy(deep=True)

    # get condition to fill on
    cond = df[col_to_fill].isnull() if replace is np.nan else df[col_to_fill] == replace

    # get data to fill with
    _temp = df[~cond].copy(deep=True)
    if method in ['mean', 'median']:
        fill = _temp.groupby(ref_col)[col_to_fill].agg(method)
    else:
        fill = _temp.groupby(ref_col)[col_to_fill].value_counts().unstack(level=-1).idxmax(axis=1)

    # fill values on dataframe
    df.loc[cond, col_to_fill] = df.loc[cond, ref_col].replace(fill)
    return df[col_to_fill], fill


def check_db(db_left, db_right, col_left, col_right=None):
    """Checks which entries of one column of a pandas DataFrame exist in a column of other pandas DataFrame
    example: which values of col_right in db_right are in col_left in db_left?

    :param db_left: pandas DataFrame with values to check
    :param db_right: pandas DataFrame with values to be checked
    :param col_left: column to check. if no col_right is passed the function assumes both databases have the same
    column name
    :param col_right: column to be checked
    :return: exists, not_exists
        exists contains the lines of db_right in which the values of col_right exist in db_left col_left
        not_exists contains the lines of df_left
    """
    if col_right is None:
        col_right = col_left

    exists = db_right[db_right[col_right].isin(db_left[col_left])]
    not_exists = db_left[~db_left[col_left].isin(db_right[col_right])]

    return exists, not_exists


def check_h5_size_and_repack(file, size=1024, compression='blosc:lz4hc', compression_factor=9, fletcher32=True,
                             verbose=0):
    """
    Checks a hdf5 file and if its size is bigger than a defined number it trys to repack it
    :param string file: path of the file to be checked and repacked
    :param int size: default: 1024. Limit of the size of the file in MB
    :param string compression: type of compression to use. Check pandas.DataFrame.to_hdf for valid compressions
    :param int compression_factor: intensity of compression (0-9). 0 = no compression
    :param bool fletcher32: whether to try and convert columns to fletcher type
    :param int verbose: param to control verbosity
    :return:
    """

    assert compression in ['zlib', 'lzo', 'bzip2', 'blosc', 'blosc:blosclz', 'blosc:blosclz4', 'blosc:blosclz4hc',
                           'blosc:snappy', 'blosc:zlib', 'blosc:zstd']
    assert compression_factor in range(0, 10)

    _current_size = os.path.getsize(file) >> 20
    if _current_size > size:
        printv('Repacking file...', level=2, verbose=verbose)
        with pd.HDFStore(file, 'r', compression_factor, compression, fletcher32) as f:
            groups = {key: f[key] for key in f}

        with pd.HDFStore(file, 'w', compression_factor, compression, fletcher32) as f:
            for key in groups:
                f[key] = groups[key]

        printv('Done', level=2, verbose=verbose)
