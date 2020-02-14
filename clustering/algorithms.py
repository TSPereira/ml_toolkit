import scipy as sp
import scipy.sparse
from kmodes.util.dissim import matching_dissim
from kmodes.kmodes import KModes
from

# todo
#  Create wrapper classes for different methods
#  Each class should save results in same attributes and have their own checks for arguments

# todo
#  keep:
#     - _check_method
#     - _check_metrics
#     - _auto_feature_selector
#     - _auto_hyperparameter_evaluator
#     - _eval_hyperparameters
#     - _run_hyperparameters
#     - _get_best_model
#     - _predictor
#     - _plot_metrics
#     - get_data_with_labels
#     - get_used_data (if auto_feature_selector is active filter original data)
#     - fit
#     - predict
#     - fit_predict
#     - save/load

from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from kmodes.kmodes import KModes


class KModes_(KModes):
    def __init__(self, n_clusters=8, max_iter=100, cat_dissim=matching_dissim, init='Cao', n_init=1, verbose=0,
                 random_state=None, n_jobs=1):
        super(KModes_, self).__init__(n_clusters, max_iter, cat_dissim, init, n_init, verbose, random_state, n_jobs)
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y=None, **kwargs):
        super(KModes_, self).fit(X, y, **kwargs)
        self.cluster_centers_ = self.cluster_centroids_
        self.inertia_ = self.cost_


class Cluster:
    _clustering_algorithms = {'kmeans': KMeans,
                              'minibatchkmeans': MiniBatchKMeans,
                              'kmodes': KModes_}

    def __init__(self, method='minibatchkmeans', verbose=0, random_state=0, **kwargs):
        self.method_hyperparams = self._get_std_params(method)
        self.model = self._clustering_algorithms[method](verbose=verbose, random_state=random_state, **kwargs)

    @classmethod
    def _get_std_params(cls, method):
        """Defines standard parameters depending on the method chosen for clustering

        :return dict: dictionary with parameter names as keys and their respective values
        """

        cls._check_method(method)
        if method in ['kmeans', 'minibatchkmeans', 'kmodes', 'kprototypes']:
            _std_params = {'min_n_clusters': 1, 'max_n_clusters': 30}
        elif method in ['DBSCAN']:
            _std_params = {'min_eps': 0.05, 'max_eps': 5, 'nr_steps_eps': 10, 'min_samples_min': 10,
                           'min_samples_max': 100, 'nr_steps_min_samples': 10}
        else:
            raise IOError('Method introduced has no standard hyperparameters to explore. '
                          'Input parameters to explore into \'self.method_params\'')
        return _std_params

    @classmethod
    def _check_method(cls, method):
        """Verifies if the method introduced by the user is implemented

        :return: None
        """

        if method not in cls._clustering_algorithms:
            raise AssertionError(
                'Method introduced is not supported. Choose one from {}'.format(cls._clustering_algorithms))


    def _auto_feature_selector(self):
        """This method can be overriden by dependant class to use a custom automatic feature selector

        As a standard this class uses a Gini Impurity Feature Selector, based on the values of Gini Impurity
        between the features and removing iteratively by the higher values observed.

        Any superseeding class must implement a method 'subset()' returning the next subset of features and an
        attribute 'current_subset_id' which identifies the subset further on.

        In the case of the GiniFeatureSelector here implemented the 'current_subset_id' identifies the gini value
        use as threshold to obtain the current subset being yielded

        :return: None
        """
        self.feature_selector = GiniFeatureSelector(pd.DataFrame(self.encoded_data.toarray()), self.n_feature_models,
                                                    verbose=self.verbose)

    def _auto_hyperparameter_evaluator(self):
        """This method can be overriden by dependant class to use a custom automatic feature selector

        As a standard this class uses a method named Cross Projection Normalization to evaluate the combinations
        of feature subsets and resulting labels

        Any superseeding class must implement a method 'evaluate()' returning a tuple of
        (optimal_subset, optimal_hyperparameter) and a method 'add_data()' to append a dictionary of format
        {subset_id: (clustering_data, labels_for_all_clusters_ran)}

        :return: None
        """

        # must have a add_data method and an evaluate method. Must return subset of best data and best params
        self.hyperparameter_evaluator = CrossProjection(verbose=self.verbose)


    def _eval_hyperparameters(self):
        """Method that depending on the clustering method chosen will evaluate and chose which are the best parameters
        for the model (from the ones tested)

        :return: None
        """

        if self.auto_feature_selection:
            self.final_model_data, _params = self.hyperparameter_evaluator.evaluate()

            if self.method in ['kmeans', 'minibatchkmeans', 'kmodes', 'kprototypes']:
                self.n_clusters = _params
                self.best_params['n_clusters'] = self.n_clusters

            elif self.method == 'DBSCAN':
                _params = _params.split(' | ')
                self.best_params['eps'], self.best_params['min_samples'] = [float(x) for x in _params]

        else:
            if self.method in ['kmeans', 'minibatchkmeans', 'kmodes', 'kprototypes']:
                self.n_clusters = eval_cluster_fitness(self.metrics_results[:, 0], dim=self.encoded_data.shape)
                self.best_params['n_clusters'] = self.n_clusters  # todo find best way to evaluate DBSCAN


    def _run_hyperparameters(self, data):
        """Method that evaluates the model for all the hyperparameters chosen and finally calls the method that will
        choose the best hyperparameters seen ('_eval_hyperparameters')

        :param np.ndarray data: encoded data array to pass to the clustering model
        :return: None
        """

        scores, key, hyper_results = run_hyperparameters(sp.sparse.csr_matrix(data), self.method, self.method_params,
                                                         self.metrics, verbose=self.verbose)

        if self.auto_feature_selection:
            _subset_id = self.feature_selector.current_subset_id
            self.metrics_results[_subset_id] = scores
            self._plot_metrics(scores, key, name=_subset_id)
            self.hyperparameter_evaluator.add_data({_subset_id: hyper_results})

            try:
                _data = self.feature_selector.subset()
                self._run_hyperparameters(_data)
            except StopIteration:
                self._eval_hyperparameters()

        else:
            self.metrics_results = scores
            self._plot_metrics(scores, key)
            self._eval_hyperparameters()
            self.final_model_data = data


    def _get_best_model(self):
        """Method that evaluates the model with the best hyperparameters found previously (or passed by the user)

        :return: None
        """

        _, _, _, self.model = clustering(sp.sparse.csr_matrix(self.final_model_data), method=self.method,
                                         verbose=self.verbose, **self.best_params)