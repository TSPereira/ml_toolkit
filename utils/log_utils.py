from time import time
from decorator import decorator
from contextlib import contextmanager


def print_progress_bar(iteration, total, prefix='', suffix='', level=1, verbose=0):
    """Prints a progress bar

    :param int iteration: Current iteration number
    :param int total: Total number of iterations
    :param string prefix: String to be used as prefix to the progress bar. if '', the word 'Progress' will be used
    :param string suffix: String to be used as suffix to the progress bar.
    :param int level: Which level of detail does the progress bar belongs too
    :param int verbose: Value to control verbosity. Progress bar will only be displayed if verbose >= level
    :return:

    Examples:
    >>>for i in range(10):
    >>>     print_progress_bar(i, 10, verbose=1)
    >>>Progress |##############################| 100.0% 10/10

    >>>for i in range(10):
    >>>     print_progress_bar(i, 10, verbose=1, prefix='Some Prefix', suffix='Some Suffix')
    >>>Some Prefix |##############################| 100.0% 10/10 Some Suffix
    """

    if int(verbose) >= level:
        iteration += 1
        if prefix == '':
            prefix = 'Progress'
        decimals = 1
        length = 30
        fill = '#'

        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        print('\r', end='')
        print(2 * (level - 1) * ' ' + '%s |%s| %s%% %s/%s %s' % (prefix, bar, percent, iteration, total, suffix),
              end='', flush=True)
        # Print New Line on Complete
        if iteration == total:
            print()

    return


def printv(str_, verbose=1, level=1, ident=0, title=False, **kwargs):
    """Prints to console dependent on verbosity level

    :param string str_: string to print
    :param int verbose: default: 1. Parameter to control verbosity
    :param int level: default: 1. Level of verbosity of the string to be printed. String will only be printed if
    the verbosity level passed is equal or higher than the string level (verbose >= level)
    :param int ident: default: 0. level of identation required for string. 2 * '' * ident
    :param bool title: default: False. Whether the string should be considered a title
    :param kwargs: additional parameters to be passed to the print function
    :return: None

    Examples:
    >>>printv('Test', level=2, verbose=1)
    >>>

    >>>printv('Test', level=2, verbose=2)
    >>>Test

    >>>printv('Test', level=2, verbose=2, ident=1)
    >>>  Test

    >>>printv('Test', level=2, verbose=2, ident=2)
    >>>    Test

    >>>printv('Test', level=2, verbose=2, ident=2, title=True)
    >>>
    >>>Test
    >>>----
    """

    if int(verbose) >= level:
        if title:
            str_ = '\n' + str_.capitalize() + '\n' + '-' * len(str_) + '\n'
        print(2 * ident * ' ' + str_, **kwargs)

    return


def timeit(verbose: bool = True):
    """Decorator to time a function

    :param bool verbose: parameter to control whether to print the timing
    :return: original function result

    Example:
    >>>@timeit(True)
    >>>def custom_sum(a, b):
    >>>     return a+b
    >>>
    >>>func(1,2)
    >>>func:custom_sum took: 0.0000s
    >>>3

    TODO: add option to change print to a log file
    """

    @decorator
    def wrap(f, *args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        if verbose:
            print(f'func:{f.__name__} took: {time()-ts:2.4f}s')
        return result
    return wrap


@contextmanager
def catchtime(verbose: bool = True, description: str = '') -> None:
    """Context Manager to time blocks of code. Everything within the with statement will be included in the timing

    :param bool verbose: parameter to control whether to print the timing
    :param str description: Descriptive of the code block
    :return:

    Example:
    >>>with catchtime(True):
    >>>     print(sum((2, 2)))
    >>>
    >>>sum((1, 2))
    >>>4
    >>>block: took: 0.0000s
    >>>3

    >>>with catchtime(True, 'Sum 2+2'):
    >>>     print(sum((2, 2)))
    >>>
    >>>sum((1, 2))
    >>>4
    >>>block:Sum 2+2 took: 0.0000s
    >>>3
    """

    start = time()
    yield

    if verbose:
        print(f'block:{description} took: {time()-start:2.4f}s')

    return
