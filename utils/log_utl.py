import sys
from time import time
from contextlib import ContextDecorator, redirect_stdout

from .os_utl import check_types


class timeit(ContextDecorator):
    """
    Decorator class that also works as context manager (with statements) to time functions or blocks of code
    It is also possible to redirect the output of the timings to a file instead of sys.stdout

    :param str|None desc: Descriptive of the function or code block
    :param int verbose: parameter to control whether to print or not.
    :param int verbose_level: level from which to print messages. verbose needs to be higher than this value for
    anything to be printed
    :param stdout: stream to print to. Can be any stream or a path to a file

    Examples:
    >>>@timeit()
    >>>def func(a, b)
    >>>     ...

    >>> with timeit():
    >>>     ...

    >>>from functools import partial
    >>>timeit = partial(timeit, verbose=1, verbose_level=1, stdout='log.txt')
    >>>
    >>>@timeit()
    >>>def func(a, b)
    >>>     ...
    """

    @check_types(verbose=int, desc=(str, type(None)))
    def __init__(self, verbose: int = 0, verbose_level: int = 1, desc: str = None, stdout=sys.stdout) -> None:
        self.verbose = verbose
        self.verbose_level = verbose_level
        self.desc = desc
        self.stdout = stdout

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, *exc):
        if self.verbose >= self.verbose_level:
            # if regular stdout just print
            # elif is already a stream (any object with write method), redirect to it
            # elif is str, open a file with str name and write to file
            if self.stdout == sys.stdout:
                self._print()

            elif hasattr(self.stdout, 'write'):
                with redirect_stdout(self.stdout):
                    self._print()

            elif isinstance(self.stdout, str):
                with open(self.stdout, 'w') as _stdout:
                    with redirect_stdout(_stdout):
                        self._print()

    def __call__(self, *args, **kwargs):
        # If used as a decorator, retrieve the function name
        func = super().__call__(*args, **kwargs)
        self._fname = func.__qualname__
        return func

    def _print(self):
        # find if used as decorator or contextmanager and assign prefix and description accordingly
        is_func = hasattr(self, '_fname')
        _prefix = 'func' if is_func else 'block'
        _desc = self.desc if self.desc is not None else (self._fname if is_func else '')
        print(f'[{_prefix}] >>> {_desc} took: {time() - self.start_time:2.4f}s')


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', level: int = 1,
                       verbose: int = 0) -> None:
    # noinspection PyUnresolvedReferences
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


def printv(str_: str, verbose: int = 1, level: int = 1, ident: int = 0, title: bool = False, **kwargs) -> None:
    # noinspection PyUnresolvedReferences
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
