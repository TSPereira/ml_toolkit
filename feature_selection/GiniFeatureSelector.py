import numpy as np
import pandas as pd
from ..utils.log_utils import print_progress_bar, printv


class GiniFeatureSelector(object):
    """
    todo write docstring after refactoring
    :param data:
    :param number_of_models:
    :param verbose:
    :param progress_bar:
    """

    def __init__(self, data, number_of_models, verbose=False, progress_bar=True):
        self.data = data
        self.verbose = verbose
        self.progress_bar = progress_bar

        if (number_of_models is not int) and (number_of_models < 1):
            raise TypeError('"number_of_models" must be a positive integer')
        self.number_of_models = number_of_models

        self.gini_matrix = self._get_gini_matrix()
        self.threshold_range = sorted(np.unique(self.gini_matrix.values.flatten()), reverse=True)
        self.filter_generator = self._get_filter_generator()
        self.current_subset_id = 'original dataset'
        self.current_n_features = data.shape[1]
        self.current_data = self.data.copy()

    def _get_filter_generator(self):
        # todo refactor
        data = pd.DataFrame(self.data).copy()

        printv('Checking all gini values... ', self.verbose)

        for model in range(self.number_of_models):
            printv('Selecting features', verbose=self.verbose, level=2)

            range_of_features = range(self.gini_matrix.shape[0])
            features_to_drop = []

            for i in range_of_features:
                for j in range_of_features:
                    if i < j:
                        if self.gini_matrix.loc[i, j] >= self.threshold_range[model]:
                            features_to_drop.append(data.columns[i])

            data[features_to_drop] = 0

            printv('> Dropped ' + str(len(set(features_to_drop))) + ' features', self.verbose, level=2)

            yield data.values, self.threshold_range[model], data.shape[1] - len(set(features_to_drop))

    def _get_gini_matrix(self, return_df=True):
        """
        Calculates Gini Matrix.
        Data is normalized to have a similar interpretation to correlation:
            0 gini impurity -> 1
            0.5 gini impurity -> 0
        :param return_df: True,returns DataFrame
        :param progress_bar: True, shows progress bar
        :return: Gini Matrix or Dataframe
        """
        # todo refactor and speed up

        printv('Calculating Gini Impurity Matrix', verbose=self.verbose)

        if type(self.data) is pd.DataFrame:
            features_matrix = self.data.values.T
        else:
            features_matrix = self.data.T

        n = self.data.shape[1]
        le = self.data.shape[0]
        matrix = np.zeros((n, n)) + np.diag(np.ones(n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    matrix[i, j] = 1 - (((features_matrix[i] != features_matrix[j]).sum()) / le) * (
                                1 - (((features_matrix[i] != features_matrix[j]).sum()) / le)) * 4

            print_progress_bar(iteration=i, total=n, verbose=self.progress_bar)

        if type(self.data) is pd.DataFrame and return_df:
            printv('> Returned Dataframe with shape: ' + str(n) + ' by ' + str(n), self.verbose, level=2)
            features_headers = self.data.columns
            gini_matrix = pd.DataFrame(matrix, columns=features_headers, index=features_headers)
            gini_matrix.fillna(0.0, inplace=True)
        else:
            printv('> Returned Array with shape: ' + str(n) + ' by ' + str(n), self.verbose, level=2)
            gini_matrix = matrix
        return gini_matrix

    def subset(self):
        """
        Wrapper method that returns the next subset of features to evaluate
        """

        _prev_n_features = self.current_n_features
        self.current_data, self.current_subset_id, self.current_n_features = self.filter_generator.__next__()

        if self.current_n_features == _prev_n_features:
            return self.subset()
        else:
            return self.current_data
