import unidecode
import re
import pandas as pd
from itertools import chain
from gensim.utils import lemmatize
from nltk.corpus import stopwords


def clean_text(raw_review, unwanted_words=None, lang='portuguese'):
    """Todo Tiago Barroca tenta generalizar. eu adicionei as unwantedwords e lang como argumentos

    :param raw_review:
    :param unwanted_words:
    :param lang:
    :return:
    """

    review_decoded = unidecode.unidecode(raw_review[0])
    review_sentences = text_to_sentences(review_decoded)

    review_lemma = []
    for sentence in review_sentences:
        review_lemma.extend(lemmatize(sentence))

    aux = []
    for word in review_lemma:
        aux.append(str(word[:-3]))

    aux_ = u' '.join(aux)

    letters_only = re.sub("[^a-zA-Z]", " ", aux_)
    no_comment = re.sub(r'\[.*?\]', " ", letters_only)

    words = no_comment.lower().split()
    stops = set(stopwords.words(lang))
    meaningful_words = [w for w in words if w not in stops if w not in unwanted_words]

    return ' '.join(meaningful_words)


def get_root_word(word):
    """Finds the root word of the passed word
    # todo não tem de haver uma definição da lingua?

    :param string word: word to identify the root of
    :return string: root of word passed
    """

    return str(lemmatize(word)[:-3])


def replace_multiple_substrings(x, list_substrings, rep_value):
    """Function to replace multiple substrings with the same value in a pandas Series or DataFrame

    :param x: pandas Series or DataFrame
    :param list_substrings: list with substrings to replace
    :param rep_value: value to apply as replacement
    :return: input x with replacements done
    """

    _replacements = {rep: rep_value for rep in list_substrings}
    return x.replace(_replacements, regex=True)


def levenshtein_distance(a, b):
    """Computes the levenshtein distance between two strings

    Todo assess speed vs levenshtein

    :param string a: string to compare
    :param string b: string to compare
    :return: minimum number of changes between both strings
    """

    if not a:
        return len(b)
    if not b:
        return len(a)
    return min(levenshtein_distance(a[1:], b[1:]) + (a[0] != b[0]), levenshtein_distance(a[1:], b) + 1,
               levenshtein_distance(a, b[1:]) + 1)


def levenshtein(s1, s2):
    """Todo assess speed vs levenshtein_distance

    :param string s1: string to compare
    :param string s2: string to compare
    :return: minimum number of changes between both strings
    """

    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def remove_words(str_or_series, list_words):
    """Remove list of words passed from the string or pandas Series provided
    It is considered a word everything that is between "spaces", ".", ";" or ","

    :param str|pandas.Series str_or_series: string or pandas Series to be cleaned
    :param list list_words: list of words to remove
    :return: cleaned string or pandas.Series
    """

    reg_dict = {r'(\s|^|[.;,]){}(\s|$|[.;,])'.format(k): r'' for k in chain.from_iterable(list_words)}

    if isinstance(str_or_series, pd.Series):
        return str_or_series.replace(reg_dict, regex=True)
    elif isinstance(str_or_series, str):
        return pd.Series([str_or_series]).replace(reg_dict, regex=True).values[0]


def remove_symbols(str_or_series, str_of_symbols):
    """Remove symbols from string or pandas Series of strings

    :param str|pandas.Series str_or_series: string or pandas Series to be cleaned
    :param str str_of_symbols: string containing the symbols to remove. Symbols passed should not be separated
    :return: cleaned string or pandas.Series
    """

    if isinstance(str_or_series, pd.Series):
        return str_or_series.str.translate(str.maketrans('', '', str_of_symbols))
    elif isinstance(str_or_series, str):
        return pd.Series([str_or_series]).str.translate(str.maketrans('', '', str_of_symbols)).values[0]


def remove_style(str_or_series, style='snake_case', fill=' '):
    """Remove styles from string or pandas Series of strings

    :param str|pandas.Series str_or_series: string or pandas Series to be cleaned
    :param str style: string that define the style to clean. default: 'snake_case'. One of {'snake_case', 'camel_case'}
    :param str fill: string to use as separator of words. default: ' '
    :return: cleaned string or pandas.Series
    """

    __styles__ = ['snake_case', 'camel_case']
    assert style in __styles__, f'"style" must be one of {__styles__}'

    pattern = fill
    if style == 'snake_case':
        pattern = r'(?<=[a-zA-Z])_(?=[a-zA-Z])'
    elif style == 'camel_case':
        pattern = r'(?<=[a-z])(?=[A-Z].)'

    if isinstance(str_or_series, pd.Series):
        return str_or_series.str.replace(pattern, fill)
    elif isinstance(str_or_series, str):
        return pd.Series([str_or_series]).str.replace(pattern, fill).values[0]


def text_to_sentences(str_or_series):
    """Separates text into its sentences. Text can be a string or a pandas Series of strings

    :param str|pandas.Series str_or_series: string or pandas Series to be separated
    :return: separated string or pandas.Series
    """

    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    if isinstance(str_or_series, pd.Series):
        return str_or_series.str.split(pattern)
    elif isinstance(str_or_series, str):
        return pd.Series([str_or_series]).str.split(pattern).values[0]


def convert_spaces(series_or_str, symbol='___', revert=False):
    old = r'\s+'
    if revert:
        old, symbol = symbol, ' '

    if isinstance(series_or_str, pd.Series):
        return series_or_str.str.replace(old, symbol)
    elif isinstance(series_or_str, str):
        return re.sub(old, symbol, series_or_str)
    else:
        raise TypeError('"series_or_str" must be of type pd.Series or string')
