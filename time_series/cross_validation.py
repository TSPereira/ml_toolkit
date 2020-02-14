import numpy as np
import warnings

from utils.os_utils import check_types


class TimeSeriesSplit:
    _vars_to_check = ['min_train_size', 'test_size']


    @check_types(n_splits=int, min_train_size=float, test_size=float, gap=float, period=(float, type(None)),
                 allow_overlap=bool, expanding=bool)
    def __init__(self, n_splits, min_train_size=0.3, test_size=0.1, gap=0.0, period=None, allow_overlap=False,
                 expanding=True):
        self.n_splits, self.min_train_size, self.test_size, self.gap, self.period = self._check_args(n_splits,
                                                                                                     min_train_size,
                                                                                                     test_size, gap,
                                                                                                     period,
                                                                                                     allow_overlap)

        self.train_size = min_train_size
        self.gap_n_samples = None
        self.allow_overlap = allow_overlap
        self.expanding = expanding


    @classmethod
    def _check_args(cls, n_splits, min_train_size, test_size, gap, period, allow_overlap):
        assert n_splits > 0, f'n_splits must be a positive integer. {n_splits} passed.'
        assert 0 <= gap < 1, f'gap must be between 0 and 1. "{gap}" passed.'
        _vars_to_check = cls._vars_to_check if period is None else cls._vars_to_check + ['period']
        for _var in _vars_to_check:
            assert 0 < eval(_var) < 1, f'{_var} must be between 0 and 1. "{eval(_var)}" passed.'

        max_test_size = np.round(1 - min_train_size - gap, 2)
        if max_test_size < 0:
            raise ValueError(f'min_train_size ({min_train_size}) + gap ({gap}) > 1 doesn\'t allow for '
                             f'test_size > 0. Try reducing one of the sizes.')

        if max_test_size < test_size:
            warnings.warn(f'Test size allowed ({max_test_size}) is lower that test_size passed ({test_size}).'
                          f'test_size was updated to maximum allowed.', stacklevel=2)
            test_size = max_test_size

        if period is None:
            period = np.round((1 - (min_train_size + gap + test_size)) / (n_splits - 1), 3)

        if not allow_overlap and (period < test_size):
            raise ValueError(f'Current period ({period}) < test_size ({test_size}). '
                             f'This will lead to overlap in consecutive validation sets, '
                             f'but allow_overlap is set to False. Either change input ratios or allow_overlap.')

        max_n_splits = int(np.round((1 - (min_train_size + gap + test_size)) / period + 1, 3))
        if max_n_splits < n_splits:
            warnings.warn(f'Maximum number of splits possible with current ratios is {max_n_splits}. '
                          f'Output will only consider {max_n_splits} folds.', stacklevel=2)
            n_splits = max_n_splits

        return n_splits, min_train_size, test_size, gap, period


    def _get_ratios(self, length):
        min_samples = np.array((self.min_train_size, self.test_size, self.period)) * length
        _check = min_samples < 1
        if any(_check):
            warnings.warn(f'Some ratio(s) requested are smaller than a single sample ({min_samples[_check]}).'
                          f'Ratio(s) will be adjusted to represent a single sample.')
        min_samples = np.clip(min_samples, a_min=1, a_max=None)

        min_samples = min_samples.astype(int)
        if min_samples[2] < min_samples[1]:
            warnings.warn('Period to add at each iteration to trainset is smaller than test_size. This will cause '
                          'overlap between testsets on consecutive folds.', stacklevel=2)

        self.gap_n_samples = max(1, int(np.round(self.gap * length))) if self.gap > 0 else 0
        initial_train_size = max(
            int(length - (min_samples[1] + self.gap_n_samples + min_samples[2] * (self.n_splits - 1))), min_samples[0])
        if initial_train_size < min_samples[0]:
            warnings.warn(f'Calculated initial_train_size ({initial_train_size}) < min_train_size allowed '
                          f'({self.min_train_size}) due to input shape and minimum values for test_size, period '
                          f'and gap')

        self.train_size = np.round(initial_train_size / length, 2)
        return initial_train_size, min_samples[1], min_samples[2]


    def get_n_splits(self, **kwargs):
        return self.n_splits


    def split(self, X, **kwargs):
        initial_train_size, test_size, period = self._get_ratios(length=X.shape[0])
        data_index = np.arange(X.shape[0])

        for i in range(self.n_splits):
            train = data_index[0 if self.expanding else period * i:initial_train_size + period * i]
            test = data_index[initial_train_size + self.gap_n_samples + period * i:
                              initial_train_size + self.gap_n_samples + period * i + test_size]

            yield train, test


class BlockTimeSeriesSplit(TimeSeriesSplit):
    """
    :param dependent_feature: Feature in data to be passed to be used to order the data and split it. If no feature
    is passed or if data passed is an array, a range of array length will be considered (thus each sample will be
    considered to have its own timestamp)
    :param n_splits: number of splits to obtain. The final number of splits might be smaller if min_train_size and
    periods passed don't allow for the entire number.
    :param min_train_size: minimum ratio of dependant_feature to be used for training
    :param test_size: ratio of dependant_feature to be considered for each test fold. If last fold has less samples,
    then the n_splits will be changed and the train_size will be increased such that all of the dataset is used.
    :param period: ratio of dependant_feature to add at each iteration to the trainset. If period passed is None,
    period will be set equal to test_size (such that there is no overlap in consecutive testsets).
    """


    @check_types(n_splits=int, min_train_size=float, test_size=float, gap=float, period=(float, type(None)),
                 allow_overlap=bool, expanding=bool)
    def __init__(self, n_splits, dependent_feature=None, min_train_size=0.3, test_size=0.1, gap=0.0, period=None,
                 allow_overlap=False, expanding=True):
        super(BlockTimeSeriesSplit, self).__init__(n_splits, min_train_size=min_train_size, test_size=test_size,
                                                   gap=gap, period=period, allow_overlap=allow_overlap,
                                                   expanding=expanding)
        self.dependent_feature = dependent_feature


    def get_n_splits(self, **kwargs):
        return self.n_splits


    def split(self, X, **kwargs):
        block_values = list(range(len(X))) if self.dependent_feature is None else X[self.dependent_feature].values
        dep_feat = np.unique(block_values)
        for x_idx, y_idx in super(BlockTimeSeriesSplit, self).split(dep_feat):
            if self.dependent_feature is None:
                yield x_idx, y_idx
            else:
                yield np.where(np.isin(block_values, dep_feat[x_idx]))[0], \
                      np.where(np.isin(block_values, dep_feat[y_idx]))[0]


if __name__ == '__main__':
    # Test for TimeSeriesSplit
    a = np.array(list(range(100)))
    b = TimeSeriesSplit(3, gap=0.1)
    print(list(b.split(a)))

    # Test for BlockTimeSeriesSplit
    import pandas as pd

    a = pd.DataFrame({'value': [1, 1, 17, 1, 2, 2, 22, 20, 4, 5, 6, 6, 8, 1, 2, 3]})
    b = BlockTimeSeriesSplit(dependant_feature='value', n_splits=4, min_train_size=0.5, test_size=0.1, period=0.2)
    print(list(b.split(a)))

    # as array
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18]]).T
    b = BlockTimeSeriesSplit(3)
    print(list(b.split(a)))
