import numpy as np
import pandas as pd
from ..feature_selection.scatter_separability_criterion import scatter_separability_score
from ..utils.log_utils import printv


class CrossProjection(object):
    """
    :param data: a dictionary with multiple one_hot_encoded csr_matrixes and labels
    :param verbose: parameter to control verbosity

    CrossProjection
    ---------------
    A class that implements Cross-Projection Normalization for multiple Clustering models using the scatter
    separability criteria.

    Developed by Closer Consulting.

    Description
    -----------
    Cross-Projection Normalization and scatter separability criteria based on the model presented in:
    Dy, J.G. and Brodley, C.E., 2004. Feature selection for unsupervised learning. Journal of machine learning
    research, 5(Aug), pp.845-889.

    Example
    -------

    # Use class

    d = {subset1: (encoded_data, [labels_1, labels_2, labels_k]),
         subset2: (encoded_data, [labels_1, labels_2, labels_k]),
         ...}
    cpn = CrossProjection(d)
    cpn_results = cpn.evaluate()

    ----
    """

    def __init__(self, data=dict(), verbose=False):
        # Inputs
        self.data = data
        self.verbose = verbose

        # Outputs
        self._id_list, self._cluster_list = [], []
        self._cpn_matrix = np.empty([0, 0])
        self.optimal_model = None
        self.optimal_id = None
        self.optimal_hyperparameters = None
        self.summary_table = pd.DataFrame()

    def _cross_projection_matrix(self):
        # todo refactor
        n = 0
        m = 0

        cpn_list = []

        for id_value in self.data:
            for cluster_k in self.data[id_value]['labels_k']:
                n += 1
                cpn_list.append((id_value, cluster_k))
            m += 1

        self._id_list, self._cluster_list = [], []
        for g, k in cpn_list:
            self._id_list.append(g)
            self._cluster_list.append(k)

        cpn_matrix = np.empty([n, m])

        for i in range(n):

            idx_s_1, idx_c_1 = cpn_list[i]
            s1 = np.asarray(self.data[idx_s_1]['ohe_data'].todense())
            c1 = self.data[idx_s_1]['labels_k'][idx_c_1]
            score1 = scatter_separability_score(s1, c1)

            for j in range(m):

                idx_s_2 = list(self.data.keys())[j]
                c2 = self.data[idx_s_2]['labels_k'][idx_c_1]  # idx_c_1 is selected on purpose

                if idx_s_1 == idx_s_2 or np.array_equal(c1, c2):
                    cpn_matrix[i, j] = 0
                else:
                    s2 = np.asarray(self.data[idx_s_2]['ohe_data'].todense())

                    cpn_matrix[i, j] = score1 * scatter_separability_score(s2, c1)

        optimal = np.unravel_index(np.argmax(cpn_matrix), cpn_matrix.shape)

        self._cpn_matrix = cpn_matrix
        self.optimal_id = self._id_list[optimal[0]]
        self.optimal_hyperparameters = self._cluster_list[optimal[0]]
        self.optimal_model = self.data[self.optimal_id]

        self.summary_table = pd.DataFrame(
                             {'optimal_id': self._id_list,
                              'clusters': self._cluster_list,
                              'score': self._cpn_matrix.max(axis=1)}).sort_values(by='score', ascending=False)

        _optimal_subset_features = self.optimal_model['ohe_data']

        return _optimal_subset_features, self.optimal_hyperparameters

    def add_data(self, data):
        """
        Adds a key, value entry to the data dictionary

        :param data: dict with identifier keys and set of labels calculated for that subset of features
        :return: nothing
        """

        self.data = {**self.data, **data}

    def evaluate(self):
        """
        Wrapper to perform the evaluation of the different subsets and labels

        :return: tuple with (optimal_subset_features, optimal_hyperparameters)
        """

        printv('Evaluating hyperparameters with Cross Projection Normalization...', self.verbose)
        return self._cross_projection_matrix()
