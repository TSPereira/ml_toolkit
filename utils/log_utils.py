import time


def print_progress_bar(iteration, total, prefix='', suffix='', level=1, verbose=False):
    """Prints a progress bar

    :param int iteration: Current iteration number
    :param int total: Total number of iterations
    :param string prefix: String to be used as prefix to the progress bar. if '', the word 'Progress' will be used
    :param string suffix: String to be used as suffix to the progress bar.
    :param int level: Which level of detail does the progress bar belongs too
    :param int verbose: Value to control verbosity. Progress bar will only be displayed if verbose >= level
    :return:
    """

    if int(verbose) >= level:
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
    """

    if int(verbose) >= level:
        if title:
            str_ = '\n' + str_.capitalize() + '\n' + '-' * len(str_) + '\n'
        print(2 * ident * ' ' + str_, **kwargs)

    return


def printd(str_, verbose=True):
    """Prints to console with a special format (paragraph before, upper letters and row of '-' after)

    :param string str_: string to print
    :param int verbose: Value to control verbosity.
    :return: None
    """

    if verbose:
        print('')
        print(str_.upper())
        print('-' * len(str_) + '\n')

    return


def timeit(start_time=0):
    """Creates a timer
    :param float start_time: 0 if first timer, time if following timers
    :return: current time
    """

    current = time.time()
    if start_time != 0:
        elapsed = current - start_time
        if elapsed > 60:
            print('  Time: ', time.strftime('%M:%S', time.gmtime(elapsed)))
        elif elapsed > 3600:
            print('  Time: ', time.strftime('%H:%M:%S', time.gmtime(elapsed)))
        else:
            print('  Time:', '%.3f' % elapsed, 'seconds')
        return current
    else:
        return current
