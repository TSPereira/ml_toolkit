# -*- coding: utf-8 -*-
import _io
import logging
import tracemalloc
import sys
import warnings
from time import time
from datetime import datetime
from contextlib import ContextDecorator, redirect_stdout
from operator import methodcaller
from textwrap import wrap
from typing import Optional, Tuple, Union

from .os_utl import check_types, check_options


# DEPRECATED ================================================================================================
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
    >>> @timeit()
    >>> def func(a, b)
    >>>      ...

    >>> with timeit():
    >>>     ...

    >>> from functools import partial
    >>> timeit = partial(timeit, verbose=1, verbose_level=1, stdout='log.txt')
    >>>
    >>> @timeit()
    >>> def func(a, b)
    >>>      ...
    """

    @check_types(verbose=int, desc=(str, type(None)))
    def __init__(self, verbose: int = 0, verbose_level: int = 1, desc: str = None, stdout=sys.stdout) -> None:
        self.verbose = verbose
        self.verbose_level = verbose_level
        self.desc = desc
        self.stdout = stdout

    def __enter__(self):
        warnings.warn('"timeit" decorator is superseeded by "monitor" and will be discontinued in version 0.2.0.',
                      DeprecationWarning, stacklevel=2)
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
        warnings.warn('"timeit" decorator is superseeded by "monitor" and will be discontinued in version 0.2.0.',
                      DeprecationWarning, stacklevel=2)

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


# ===========================================================================================================
class monitor(ContextDecorator):
    """Decorator class that also works as context manager (with statements) to time functions or blocks of code
    It is also possible to redirect the output of the timings to a file instead of sys.stdout

    Args:
        mode: Which mode to use. Controls the output message. Default: 'all'. Options: ['all', 'time', 'memory']
        desc: Descriptive of the function or code block
        disable_prefix: Whether to disable the prefix generated automatically
        logger: Logger to use for messages. Default: sys.stdout. Can be one of [_io.TextIOWrapper, _io.StringIO, file,
                string, instance of logging.Logger, sys.stdout or sys.stderr].

    Examples:
        >>> @monitor
        >>> def func(a, b)
        >>>      ...

        >>> with monitor():
        >>>     ...

        >>> from functools import partial
        >>> monitor = partial(monitor, mode='time', disable_prefix=True, case_style='upper')
        >>>
        >>> @monitor
        >>> def func(a, b)
        >>>      ...
    """
    @check_types(desc=(str, type(None)), disable_prefix=bool)
    @check_options(mode=('all', 'memory', 'time'))
    def __init__(self, mode='all', desc: str = None, disable_prefix: bool = False, logger=sys.stdout, log_level='info',
                 **kwargs) -> None:
        self.desc = desc
        self.disable_prefix = disable_prefix
        self.mode = mode
        self.msg = None
        self.duration = None
        self.memory = None
        self.logger = self._set_logger(logger)
        self.log_level = log_level
        self._kwargs = kwargs

    @staticmethod
    def _set_logger(logger):
        if isinstance(logger, logging.Logger):
            return logger

        elif hasattr(logger, 'write'):
            return create_logger(f'{__name__}.monitor', logger, on_conflict='override')

        elif isinstance(logger, str):
            return create_logger(f'{__name__}.monitor', None, file=logger, on_conflict='override')

        else:
            raise TypeError('"logger" must be one instance of one of [logging.Logger, _io.TextIOWrapper, str')

    def __call__(self, func):
        # If used as a decorator, retrieve the function name
        self._fname = func.__qualname__
        func = super().__call__(func)
        func.logger = self.logger
        return func

    def __enter__(self):
        self.start_time = datetime.now()
        tracemalloc.start()
        return self

    def __exit__(self, *exc):
        is_func = hasattr(self, '_fname')
        self.duration = time_diff(datetime.now(), self.start_time)
        self.memory = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 2)
        tracemalloc.stop()

        # prepare string to be printed
        _desc = self.desc or (self._fname if is_func else '')
        _prefix = 'func' if is_func else 'block'
        _prefix = f'[{_prefix}' + ('' if not _desc else f': {_desc}') + '] '

        prefix = (not self.disable_prefix) * _prefix
        time_msg = f'took: {self.duration}'
        mem_msg = f'peak: {self.memory} MB'

        msg = prefix \
                   + (self.mode in ('time', 'all')) * time_msg \
                   + (self.mode == 'all') * ' / ' \
                   + (self.mode in ('memory', 'all')) * mem_msg

        self.msg = print_formatter(msg, **self._kwargs)

        # log the message
        getattr(self.logger, self.log_level)(self.msg)


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    colors = dict(grey="\x1b[38;21m",
                  yellow="\x1b[33;21m",
                  red="\x1b[31;21m",
                  bold_red="\x1b[31;1m",
                  reset="\x1b[0m")

    FORMATS = {
        logging.DEBUG: colors['grey'] + '{msg}' + colors['reset'],
        logging.INFO: colors['grey'] + '{msg}' + colors['reset'],
        logging.WARNING: colors['yellow'] + '{msg}' + colors['reset'],
        logging.ERROR: colors['red'] + '{msg}' + colors['reset'],
        logging.CRITICAL: colors['bold_red'] + '{msg}' + colors['reset']
    }

    def __init__(self, colored=True, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self._colored = colored

    @property
    def is_colored(self):
        return self._colored

    @is_colored.setter
    @check_types(value=bool)
    def is_colored(self, value: bool):
        self._colored = value

    def format(self, record):
        if self.is_colored:
            levelno = getattr(record, 'levelno', 20)
            return self.FORMATS.get(levelno, 20).format(msg=super().format(record))
        else:
            return super().format(record)


@check_options(on_conflict=('add', 'override', 'raise', 'ignore'))
def create_logger(name: str,
                  consolehandler: Optional[Union[_io.TextIOWrapper, _io.StringIO]] = sys.stdout,
                  file: Optional[str] = None,
                  logger_level: int = logging.INFO,
                  handlers_level: Union[int, Tuple[int, ...]] = logging.DEBUG,
                  filemode: str = 'w',
                  fmt: str = '[%(asctime)s | %(levelname)s] %(message)s',
                  datefmt: str = '%Y-%m-%d %H:%M:%S',
                  on_conflict: str = 'raise',
                  colored: bool = True) -> logging.Logger:
    """Creates a logger with the logging module. It can define one or both of ConsoleHandler and FileHandler.
    ConsoleHandler will handle how messages are printed on console/terminal, while FileHandler will control how
    messages are saved into the passed file.

    Args:
        name: Name to identify the logger.
        consolehandler: Whether to add a console handler or not. Default: sys.stdout.
                        Options: [sys.stdout, sys.stderr, None]
        file: path to file to use for logging. A FileHandler will be added pointing to that file.
        logger_level: Minimum level for the logger to log any message.
        handlers_level: Minimum level(s) for the handlers to log.
        filemode: file mode in which to open a file log. Default: "w".
        fmt: Format to use on the messages logged. Follows "logging" conventions.
        datefmt: Format to use for datetime (if present in the message). Follows "logging" conventions.
        on_conflict: How to solve conflicts if a logger with the same name already exists. Default: 'raise'.
                     Options: ['raise', 'override', 'add'].
        colored: Whether to use a colored formatter (for consolehandler only)

    Returns:
        logging.Logger: generated logger

    Examples:
        >>> logger = create_logger(__name__, file=f'{__name__}.log', logger_level=logging.DEBUG,
        >>>                        handlers_level=(10, 30), fmt='%(message)s', on_conflict='override')
    """

    def set_color_for_all_color_formatters(self: logging.Logger, value):
        for h in self.handlers:
            f = h.formatter
            if isinstance(f, ColorFormatter):
                f.is_colored = value
        return

    # check for conflicts
    if name in logging.Logger.manager.loggerDict:
        if on_conflict == 'raise':
            raise KeyError(f'A logger with name "{name}" is already defined.')
        elif on_conflict == 'add':
            warnings.warn(f'\nA logger with name "{name}" is already defined. Any additional handlers will be added.\n'
                          f'This might result in duplicated messages.', UserWarning, stacklevel=2)

        elif on_conflict == 'ignore':
            warnings.warn(f'\nA logger with name "{name}" is already defined. Nothing will be updated', UserWarning,
                          stacklevel=2)
            return logging.Logger.manager.loggerDict.get(name)

        else:
            logging.Logger.manager.loggerDict.pop(name)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logger_level)
    logger.setFormattersIsColored = set_color_for_all_color_formatters.__get__(logger, logging.Logger)

    if isinstance(handlers_level, int):
        handlers_level = (handlers_level, )

    if consolehandler is not None:
        if not hasattr(consolehandler, 'write'):
            raise TypeError('"consolehandler" passed must have a "write" method.')

        ch = logging.StreamHandler(consolehandler)
        ch.setLevel(handlers_level[0])
        ch.setFormatter(ColorFormatter(colored=colored, fmt=fmt, datefmt=datefmt))
        logger.addHandler(ch)

    if file is not None:
        fh = logging.FileHandler(file, mode=filemode, encoding='utf8')
        fh.setLevel(handlers_level[-1])
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(fh)

    if not logger.handlers:
        warnings.warn('Logger created without handlers.')

    # make messages logged to this logger uniquely on this specific logger (they won't propagate to the root logger)
    logger.propagate = False
    return logger


@check_options(as_type=('tuple', 'string'))
def time_diff(end_time: datetime, start_time: datetime, as_type: str = 'string') -> Union[tuple, str]:
    """Computes the difference between two datetimes and returns it in hours, minutes, seconds and microseconds

    Args:
        end_time: endtime of the operation
        start_time: starttime of the operation
        as_type: whether to return as a parsed string or as a tuple

    Returns:
        Tuple or String: Difference of time as (hours, minutes, seconds, microseconds) or
            "[hours}h {minutes}m {seconds}s {microseconds}μs"

    Examples:
        >>> start_time = datetime(2021, 4, 17, 23, 11, 22, 829688)
        >>> end_time = datetime(2021, 4, 18, 18, 15, 22, 100)
        >>> time_diff(end_time, start_time)
        '19h 3m 59s 170412μs'

        >>> time_diff(end_time, start_time, as_type='tuple')
        (19, 3, 59, 170412)
    """
    t_diff = (end_time - start_time).total_seconds()

    # convert anything above hours into hours
    hours, rem = int(t_diff // 3600), t_diff % 3600
    minutes, rem = int(rem // 60), rem % 60
    seconds = int(rem)
    microseconds = int(round((rem - seconds) * 1e6))

    return (hours, minutes, seconds, microseconds) if as_type == 'tuple' else \
        f'{hours}h {minutes}m {seconds}s {microseconds}\u03BCs'


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', level: int = 1,
                       verbose: int = 0) -> None:
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


def printv(string: str, verbose: int = 1, verbose_level: int = 1, ident: int = 0, case_style: Optional[str] = None,
           title: bool = False, **kwargs) -> None:
    """Prints to console dependent on verbosity level

    Args:
        string: string to print
        verbose: Parameter to control verbosity. Default: 1.
        verbose_level: Level of verbosity of the string to be printed. String will only be printed if the
            verbosity level passed is equal or higher than the string level (verbose >= level). Default: 1.
        ident: Level of identation required for string. 1 level = 2 * ' ' * ident. Default: 0.
        case_style: Style to apply to string.
        title: Whether the string should be considered a title. Default: False.
        **kwargs: Additional parameters to be passed to the print function

    Returns:
        None

    Examples:
        >>> printv('Test', verbose_level=2, verbose=1)


        >>> printv('Test', verbose_level=2, verbose=2)
        Test

        >>> printv('Test', verbose_level=2, verbose=2, ident=1)
          Test

        >>> printv('Test', verbose_level=2, verbose=2, ident=2)
            Test

        >>> printv('Test', verbose_level=2, verbose=2, ident=2, title=True)
        <empty paragraph>
        Test
        ----
    """

    if int(verbose) >= verbose_level:
        print(print_formatter(string, ident, title=title, case_style=case_style), **kwargs)

    return


@check_options(case_style=(None, 'capitalize', 'title', 'upper', 'lower'))
def print_formatter(string: str, ident: int = 0, case_style=None, title: bool = False) -> str:
    """Format strings to be printed

    Args:
        string: string to be formatted
        ident: level of identation
        case_style: case to apply to string. one of [None, 'capitalize', 'title', 'upper', 'lower']. Default: None
        title: Whether to add dashes below the string or not

    Returns:
        String: string formatted
    """
    string = string if case_style is None else methodcaller(case_style)(string)

    if title:
        if case_style is None:
            string = string.title()
        string = '\n' + string + '\n' + '-' * len(string) + '\n'

    return 2 * ident * ' ' + string


def wrap_text(text: str, width: int = 80, **kwargs) -> str:
    """Wrapper to wrap text and return a string

    Args:
        text: Text to be wrapped.
        width: Width to wrap to. Default: 80
        **kwargs: additional keyword arguments for textwrap.wrap function

    Returns:
        String: String to be printed
    """
    return '\n'.join(wrap(text, width, **kwargs))
