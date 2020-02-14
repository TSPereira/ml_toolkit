import numpy as np
from scipy.stats import shapiro, normaltest, anderson, kstest, kurtosistest, skewtest, norm
from ..utils import printv


# todo UNDER DEVELOPMENT
def check_normality(arr, alpha=0.05, verbose=0):
    __tests__ = {# 'Shapiro-Wilk': shapiro,
                 'D\'Agostino\'s K^2': normaltest,
                 'Anderson-Darling': anderson,
                 'Kolmogorov-Smirnov': kstest,
                 'kurtosistest': kurtosistest,
                 'skewtest': skewtest}

    if alpha not in [0.01, 0.025, 0.05, 0.1, 0.15]:
        __tests__.pop('Anderson-Darling')

    # Get normal distribution for the size of the input
    nrm = norm.rvs(arr.shape[0])

    results = True
    for name, test in __tests__.items():
        if name == 'Shapiro-Wilk':
            stat, p = test(arr)
            printv(f'{name} Statistics={stat:.3f}, p={p:.3f}', verbose=verbose, level=1)
            results = _validation(p <= alpha, name, verbose=verbose)

        elif name in ['D\'Agostino\'s K^2', 'kurtosistest', 'skewtest']:
            stat, p = test(arr, nan_policy='omit')
            printv(f'{name} Statistics={stat:.3f}, p={p:.3f}', verbose=verbose, level=1)
            results = _validation(p <= alpha, name, verbose=verbose)

        elif name == 'Kolmogorov-Smirnov':
            stat, p = test(arr, 'norm')
            printv(f'{name} Statistics={stat:.3f}, p={p:.3f}', verbose=verbose, level=1)
            results = _validation(p <= alpha, name, verbose=verbose)

        elif name == 'Anderson-Darling':
            result = anderson(arr)
            printv(f'{name} Statistic={result.statistic:.3f}', verbose=verbose, level=1)

            _ind = np.where(result.significance_level == alpha * 100)[0][0]
            results = _validation(result.statistic >= result.critical_values[_ind], name, verbose=verbose)

        if not results:
            break

    return results


def _validation(cond, name, verbose=0):
    if cond:
        printv(f'{name} test violated. Distribution is not normal.', verbose=verbose, level=1)
        return False
    else:
        printv(f'{name} test passed successfully.', verbose=verbose, level=1)
        return True

# 'Kolmogorov-Smirnov': kstest,
# elif name == 'Kolmogorov-Smirnov':
#     stat, p = test(arr, norm.rvs(size=len(arr)))
#     printv(f'{name} Statistics={stat:.3f}, p={p:.3f}', verbose=verbose, level=1)
#     results = _validation(p <= alpha, name, verbose=verbose)