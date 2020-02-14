from calendar import monthrange
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def add_date_features(df, column_name, month_names=False, weekday_names=True):
    """Appends columns with year, month, day and weekday to a data-frame considering a column filled with datetimes.

    :param pandas.DataFrame df: DataFrame with column to work on
    :param string column_name: name of the column to use as basis to create date_features
    :param bool month_names: default: False. boolean to control whether months come as strings or as numbers
    :param bool weekday_names: default: True. boolean to control whether weekdays come as strings or as numbers
    :return: None (DataFrame is changed as reference)
    """

    # coerce will force any cell that raises an exception when converting to be represented as NaT (null for datetimes)
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name + '_year'] = df[column_name].dt.year.astype('Int16')
    df[column_name + '_month'] = df[column_name].dt.month_name() if month_names else df[column_name].dt.month.astype(
        'Int16')
    df[column_name + '_day'] = df[column_name].dt.day.astype('Int16')
    df[column_name + '_weekday'] = df[column_name].dt.weekday_name if weekday_names else df[
        column_name].dt.weekday.astype('Int16')
    return


def date_to_age(birth_date, reference_date='today', errors='raise', format_='years'):
    """Converts birth dates strings in any format to age
    If date_str not in correct format, exception will be
    raised and nothing and np.nan will be returned

    :param string birth_date: in '%Y-%m-%d' format
    :param string reference_date: date to calculate age to in '%Y-%m-%d' format
    :param string errors: default: 'raise'. One of ['raise', 'coerce']. Controls what to do in case of an exception
    raise at datetime conversion. 'coerce' will return np.nan. 'raise' will raise the exception
    :param string format_: default: 'years'. One of ['years', 'months', days']
    :return: age or np.nan if error converting dates and errors == 'coerce'
    """

    assert errors in ['raise', 'coerce']
    assert format_ in ['years', 'months', 'days']

    try:
        birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
        reference_date = datetime.today() if reference_date == 'today' else \
            datetime.strptime(reference_date, '%Y-%m-%d')

    except ValueError:
        if errors == 'raise':
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        elif errors == 'coerce':
            return np.nan

    else:
        if format_ == 'years':
            return (reference_date.year - birth_date.year) - (
                        (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
        elif format_ == 'months':
            return month_delta(birth_date, reference_date)
        elif format_ == 'days':
            return (reference_date - birth_date).days


def month_delta(date1, date2):
    """Finds the delta in months between two dates

    :param datetime date1: date
    :param datetime date2: date
    :return: difference in months between both dates
    """

    delta = 0

    if date1 > date2:
        date1, date2 = date2, date1

    while True:
        mdays = monthrange(date1.year, date1.month)[1]
        date1 += timedelta(days=mdays)
        if date1 <= date2:
            delta += 1
        else:
            break
    return delta
