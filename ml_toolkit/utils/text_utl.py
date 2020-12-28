import re
from itertools import chain
from typing import Union, Iterable, Any

from pandas import DataFrame, Series

from .os_utl import check_types, check_options


@check_types(x=(Series, DataFrame), list_substring=Iterable)
def replace_multiple_substrings(x: Union[Series, DataFrame], list_substrings: Iterable, rep_value: Any) -> \
        Union[Series, DataFrame]:
    """Function to replace multiple substrings with the same value in a pandas Series or DataFrame

    :param Series|DataFrame x: pandas Series or DataFrame to replace on
    :param Iterable list_substrings: Iterable with substrings to replace
    :param rep_value: value to apply as replacement
    :return: input x with replacements done
    """

    _replacements = {rep: rep_value for rep in list_substrings}
    return x.replace(_replacements, regex=True)


@check_types(str_or_series=(str, Series), list_substring=Iterable)
def remove_words(str_or_series: Union[str, Series], list_words: Iterable) -> Union[str, Series]:
    """Remove list of words passed from the string or pandas Series provided
    It is considered a word everything that is between "spaces", ".", ";" or ","

    :param str|Series str_or_series: string or pandas Series to be cleaned
    :param Iterable list_words: list of words to remove
    :return: cleaned string or Series
    """

    reg_dict = {r'(\s|^|[.;,]){}(\s|$|[.;,])'.format(k): r'' for k in chain.from_iterable(list_words)}

    if isinstance(str_or_series, Series):
        return str_or_series.replace(reg_dict, regex=True)
    elif isinstance(str_or_series, str):
        return Series([str_or_series]).replace(reg_dict, regex=True).values[0]


@check_types(str_or_series=(str, Series), str_of_symbols=Iterable)
def remove_symbols(str_or_series: Union[str, Series], symbols: Iterable) -> Union[str, Series]:
    """Remove symbols from string or pandas Series of strings

    :param str|Series str_or_series: string or pandas Series to be cleaned
    :param Iterable symbols: string containing the symbols to remove. Symbols passed should not be separated
    :return: cleaned string or Series
    """

    symbols = ''.join(symbols)
    if isinstance(str_or_series, Series):
        return str_or_series.str.translate(str.maketrans('', '', symbols))
    elif isinstance(str_or_series, str):
        return Series([str_or_series]).str.translate(str.maketrans('', '', symbols)).values[0]


@check_options(style=('snake_case', 'camel_case'))
@check_types(str_or_series=(str, Series), style=str)
def remove_style(str_or_series, style='snake_case', sep: str = ' ') -> Union[str, Series]:
    """Remove styles from string or pandas Series of strings

    :param str|pandas.Series str_or_series: string or pandas Series to be cleaned
    :param str style: string that define the style to clean. default: 'snake_case'. One of {'snake_case', 'camel_case'}
    :param str sep: string to use as separator of words. default: ' '
    :return: cleaned string or Series
    """

    if style == 'snake_case':
        pattern = r'(?<=[a-zA-Z])_(?=[a-zA-Z])'
    elif style == 'camel_case':
        pattern = r'(?<=[a-z])(?=[A-Z].)'
    else:
        pattern = sep

    if isinstance(str_or_series, Series):
        return str_or_series.str.replace(pattern, sep)
    elif isinstance(str_or_series, str):
        return Series([str_or_series]).str.replace(pattern, sep).values[0]


@check_types(str_or_series=(str, Series))
def text_to_sentences(str_or_series: Union[str, Series]) -> Union[str, Series]:
    """Separates text into its sentences. Text can be a string or a pandas Series of strings

    :param str|pandas.Series str_or_series: string or pandas Series to be separated
    :return: separated string or pandas.Series
    """

    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    if isinstance(str_or_series, Series):
        return str_or_series.str.split(pattern)
    elif isinstance(str_or_series, str):
        return Series([str_or_series]).str.split(pattern).values[0]


@check_types(str_or_series=(str, Series), symbol=str, revert=bool)
def convert_spaces(series_or_str: Union[str, Series], symbol: str = '___', revert: bool = False) -> Union[str, Series]:
    """Converts spaces into a symbol. Using the flag "revert" can also convert a symbol into a space

    :param str|Series series_or_str: string or Series to apply conversion on
    :param str symbol: symbol to convert to/from
    :param bool revert: Flag to control whether to convert spaces into symbol or symbol into spaces
    :return: original string/Series with the symbol converted
    """

    old = r'\s+'
    if revert:
        old, symbol = symbol, ' '

    if isinstance(series_or_str, Series):
        return series_or_str.str.replace(old, symbol)
    elif isinstance(series_or_str, str):
        return re.sub(old, symbol, series_or_str)
