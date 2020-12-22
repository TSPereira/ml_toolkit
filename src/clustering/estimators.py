import math
import numpy as np
import warnings
from collections import Counter

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

from .metrics import gap_statistic, silhouette_score_block
from ..utils.os_utl import check_types, filter_kwargs
from ..utils.log_utl import print_progress_bar, printv


"""
GapStatistic estimator is the Miles Granger implementation 2016.04.25.1430 shared in 
https://anaconda.org/milesgranger/gap-statistic/notebook
"""


class EstimatorMixin:
    def __init__(self, cluster_fn=MiniBatchKMeans, **kwargs):
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn
        self.metrics = None

    @staticmethod
    def _get_best_votes(arr):
        k_best_n = arr[arr[:, 1].argsort()[::-1], 0].astype(int)
        return k_best_n.tolist()

    @staticmethod
    def _get_max_k(x, max_k):
        if max_k > x.shape[0]:
            print(f'Can only use number of clusters lower than number of examples ({x.shape[0]}).')
        return min(max_k, x.shape[0])

    def plot_metric(self, ax=None, show=True, normalise=False, n_votes=3):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(24, 12))

        # metric
        x, y = self.metrics.T
        if normalise:
            y = MinMaxScaler().fit_transform(y.reshape(-1, 1))

        ax.plot(x, y, label=f'{self.__class__.__qualname__} ({self.best_votes[0]})', linewidth=0.7)

        # votes
        votes = np.array(self.best_votes[:n_votes])
        ax.scatter(votes, y[votes - 1], color=ax.lines[-1].get_color(), alpha=0.3, edgecolors='k',
                   s=np.array(range(25, (n_votes+1)*15 + 1, 15))[::-1])

        if show:
            ax.legend()
            plt.show()

        return

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__qualname__}'


class Riddle(EstimatorMixin):
    """
    Riddle K-estimator.
    Estimates the correct value for K using the reciprocal delta log rule.
    """

    @check_types(x=np.ndarray, max_k=int)
    def fit(self, x, max_k=50, **kwargs):
        """
        Directly fit this estimator to original data considering max_k clusters.
        Kwargs passed are not used. They are included for estimators compatibility only
        :param np.ndarray x: Array with the data to cluster
        :param int max_k: Maximum number of clusters to try to find
        :param kwargs:
        :return int: Best number of clusters
        """
        # calculate s_k
        max_k = self._get_max_k(x, max_k)
        s_k = np.array([self.cluster(k).fit(x).inertia_ for k in range(1, max_k + 1)])
        return self.fit_s_k(s_k)

    @check_types(s_k=(np.ndarray, list, tuple))
    def fit_s_k(self, s_k, **kwargs):
        """
        Fit the estimator to the distances of each datapoint to assigned cluster centroid
        Kwargs passed are not used. They are included for estimators compatibility only
        :param np.ndarray|list|tuple s_k: Collection of inertias_ for each n_cluster explored
        :param kwargs:
        :return int: Best number of clusters
        """
        if isinstance(s_k, (list, tuple)):
            s_k = np.array(s_k)

        r_k = 1/s_k
        n_cl = range(1, len(r_k) + 1)

        diff = np.pad(np.diff(r_k), (1, 0), 'constant', constant_values=-np.inf)
        results = diff/np.log(n_cl)
        results[results == -np.inf] = results[results != -np.inf].min()

        self.metrics = np.vstack((n_cl, results)).T
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K


class AsankaPerera(EstimatorMixin):
    """Estimates the K-Means K hyperparameter through geometrical analysis of the distortion curve"""

    def __init__(self, tolerance=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance

    @check_types(x=np.ndarray, max_k=int)
    def fit(self, x, max_k=50, **kwargs):
        """
        Directly fit this estimator to original data considering max_k clusters.
        :param np.ndarray x: Array with the data to cluster
        :param int max_k: Maximum number of clusters to try to find
        :return int: Best number of clusters
        """

        max_k = self._get_max_k(x, max_k)

        s_k_list = list()
        sk0 = 0
        for k in range(1, max_k + 1):
            sk1 = self.cluster(k).fit(x).inertia_
            s_k_list.append(sk1)
            if k > 2 and abs(sk0 - sk1) < self.tolerance:
                break
            sk0 = sk1

        # Pass the line endpoints and find max dist
        self.metrics = self.find_distances(np.array(s_k_list), x0=1, y0=s_k_list[0], x1=3 * len(s_k_list), y1=0)
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K

    @check_types(s_k=(np.ndarray, list, tuple))
    def fit_s_k(self, s_k, **kwargs):
        """
        Fit the estimator to the distances of each datapoint to assigned cluster centroid
        :param np.ndarray|list|tuple s_k: Collection of inertias_ for each n_cluster explored
        :param kwargs: Tolerance to be used can be passed as a kwarg
        :return int: Best number of clusters
        """

        if isinstance(s_k, list):
            s_k = np.array(s_k)

        s_k_list = list()
        sk0 = 0
        for k, sk1 in enumerate(s_k):
            s_k_list.append(sk1)
            if (k > 2) and (abs(sk0 - sk1) < self.tolerance):
                break
            sk0 = sk1

        # Pass the line endpoints and find max dist
        self.metrics = self.find_distances(np.array(s_k_list), x0=1, y0=s_k_list[0], x1=3 * len(s_k_list), y1=0)
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K

    @classmethod
    def find_distances(cls, s_k, x0, y0, x1, y1):
        """
        Find the largest distance from each point in s_k (defined by (x=position in array/list, y=value)) to line
        defined by (x0, y0) and (x1, y1)
        :param np.ndarray|list|tuple s_k: values of y of datapoints to test
        :param int|float x0: Coordinate x of point 0
        :param int|float y0: Coordinate y of point 0
        :param int|float x1: Coordinate x of point 1
        :param int|float y1: Coordinate y of point 1
        :return np.ndarray: Array with (best number of clusters, distance to line)
        """
        k_dist = np.array([[k, cls.distance_to_line(k, s_k[k - 1], x0, y0, x1, y1)] for k in range(1, len(s_k) + 1)])
        if int(k_dist[k_dist[:, 1].argmax(), 0]) == int(k_dist[-1, 0]):
            print('AsankaPerera: Number of clusters explored is not optimal! Run analysis with higher number of '
                  'clusters.')
        return k_dist

    @staticmethod
    def distance_to_line(x0, y0, x1, y1, x2, y2):
        """
        Calculates the distance from (x0,y0) to the line defined by (x1,y1) and (x2,y2)
        :param int|float x0: Coordinate x of point 0
        :param int|float y0: Coordinate y of point 0
        :param int|float x1: Coordinate x of point 1
        :param int|float y1: Coordinate y of point 1
        :param int|float x2: Coordinate x of point 2
        :param int|float y2: Coordinate y of point 2
        :return float: distance between point0 and the line defined by point1 and point2
        """
        dx = x2 - x1
        dy = y2 - y1
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx * dx + dy * dy)


class PhamDimovNguyen(EstimatorMixin):
    """Estimates the best value for K using Pham-Dimov-Nguyen method"""

    @check_types(x=np.ndarray, max_k=int)
    def fit(self, x, max_k=50, **kwargs):
        """
        Directly fit this estimator to original data considering max_k clusters.
        Kwargs passed are not used. They are included for estimators compatibility only
        :param np.ndarray x: Array with the data to cluster
        :param int max_k: Maximum number of clusters to try to find
        :param kwargs:
        :return int: Best number of clusters
        """
        max_k = self._get_max_k(x, max_k)
        s_k = np.array([self.cluster(k).fit(x).inertia_ for k in range(1, max_k + 1)])
        return self.fit_s_k(s_k, data_shape=x.shape)

    @check_types(s_k=(np.ndarray, list, tuple), data_shape=(np.ndarray, tuple, list))
    def fit_s_k(self, s_k, data_shape, **kwargs):
        """
        Fit the estimator to the distances of each datapoint to assigned cluster centroid
        Kwargs passed are not used. They are included for estimators compatibility only
        :param np.ndarray|list|tuple s_k: Collection of inertias_ for each n_cluster explored
        :param np.ndarray|list|tuple data_shape: shape of the data used to cluster
        :param kwargs:
        :return int: Best number of clusters
        """
        if isinstance(s_k, list):
            s_k = np.array(s_k)

        # calculate all alphas
        a_k = np.zeros(len(s_k) + 1)
        for i in range(2, len(s_k) + 1):
            a_k[i] = self._alpha_k(a_k, i, data_shape[1])

        # pad s_k to move them to correct cluster position
        s_k = np.pad(s_k, (1, 0), 'constant', constant_values=0)

        # evaluate function for all positions
        with np.errstate(divide='ignore'):
            f_k = s_k[1:] / (a_k[1:] * s_k[:-1])
            f_k[f_k == np.inf] = f_k[f_k != np.inf].max()
            self.metrics = np.vstack((range(1, len(f_k) + 1), f_k)).T

        # get K from where function result is minimum and if needed correct it
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K

    @staticmethod
    def _alpha_k(a_k, k, dim):
        if k == 2:
            ak = 1.0 - 3.0 / (4.0 * dim)
        else:
            ak1 = a_k[k - 1]
            ak = ak1 + (1.0 - ak1) / 6.0
        return ak

    @staticmethod
    def _get_best_votes(arr):
        _arr = arr[arr[:, 1] <= 0.85]
        k_best_n = _arr[_arr[:, 1].argsort(), 0].astype(int)

        rem = arr[~np.isin(arr[:, 0], k_best_n)]
        rem = rem[rem[:, 1].argsort(), 0].astype(int)
        return list(k_best_n) + list(rem)


class GapStatistic(EstimatorMixin):
    """
    Class that implements the GapStatistic estimator as defined by Miles Granger (implementation 2016.04.25.1430)
    shared in  https://anaconda.org/milesgranger/gap-statistic/notebook
    """

    def __init__(self, n_refs=3, **kwargs):
        super().__init__(**kwargs)
        self.n_refs = n_refs

    def fit(self, x, max_k=50, **kwargs):
        """
        Directly fit this estimator to original data considering max_k clusters.
        :param np.ndarray x: Array with the data to cluster
        :param int max_k: Maximum number of clusters to try to find
        :param kwargs: additional arguments for cluster function passed, e.g: batch_size
        :return int: Best number of clusters
        """

        max_k = self._get_max_k(x, max_k)
        results = list()
        for gap_index, k in enumerate(range(1, max_k + 1)):
            # fit data to cluster function provided and Calculate gap statistic
            orig_disp = self.cluster(k, **kwargs).fit(x).inertia_
            results.append((k, gap_statistic(k, orig_disp, x.shape, self.n_refs, **kwargs)))

        self.metrics = np.array(results)
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K

    def fit_s_k(self, s_k, data_shape, **kwargs):
        """
        Fit the estimator to the distances of each datapoint to assigned cluster centroid
        :param np.ndarray|list|tuple s_k: Collection of inertias_ for each n_cluster explored
        :param np.ndarray|list|tuple data_shape: shape of the data used to cluster
        :param kwargs: additional arguments for cluster function passed, e.g: batch_size
        :return int: Best number of clusters
        """

        self.metrics = np.array([(k, gap_statistic(k, s_k[gap_index], data_shape, self.n_refs, **kwargs))
                                 for gap_index, k in enumerate(range(1, len(s_k) + 1))])
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K


class Silhouette(EstimatorMixin):
    def fit(self, x, max_k=50, **kwargs):
        # Filter kwargs
        kwargs_ = filter_kwargs(kwargs, self.cluster.__init__)

        # calculate s_k
        max_k = self._get_max_k(x, max_k)
        labels = [self.cluster(k, **kwargs_).fit(x).labels_ for k in range(1, max_k + 1)]
        return self.fit_s_k(x, labels, **kwargs)

    def fit_s_k(self, data: np.ndarray, labels: list, **kwargs):
        # Filter kwargs
        kwargs = filter_kwargs(kwargs, silhouette_score_block)

        # Compute silhouette score for each set of labels
        with np.errstate(divide='ignore', invalid='ignore'):
            results = [(i, silhouette_score_block(data, lbls, **kwargs)) for i, lbls in enumerate(labels, 1)]

        # Find the best votes
        self.metrics = np.array(results)
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K


class CalinskiHarabasz(EstimatorMixin):
    def fit(self, x, max_k=50, **kwargs):
        # Filter kwargs
        kwargs_ = filter_kwargs(kwargs, self.cluster.__init__)

        # calculate s_k
        max_k = self._get_max_k(x, max_k)
        labels = [self.cluster(k, **kwargs_).fit(x).labels_ for k in range(1, max_k + 1)]
        return self.fit_s_k(x, labels)

    def fit_s_k(self, data: np.ndarray, labels: list, **kwargs):
        # Compute score for each set of labels
        results = []
        for i, lbls in enumerate(labels, 1):
            try:
                results.append((i, calinski_harabasz_score(data, lbls)))
            except ValueError:
                results.append((i, float('-inf')))
        else:
            self.metrics = np.array(results)
            self.metrics[self.metrics[:, 1] == -np.inf, 1] = 0

        # Find the best votes
        self.best_votes = self._get_best_votes(self.metrics)
        self.K = self.best_votes[0]
        return self.K


class DaviesBouldin(EstimatorMixin):
    def fit(self, x, max_k=50, **kwargs):
        # Filter kwargs
        kwargs_ = filter_kwargs(kwargs, self.cluster.__init__)

        # calculate s_k
        max_k = self._get_max_k(x, max_k)
        labels = [self.cluster(k, **kwargs_).fit(x).labels_ for k in range(1, max_k + 1)]
        return self.fit_s_k(x, labels)

    def fit_s_k(self, data: np.ndarray, labels: list, **kwargs):
        # Compute score for each set of labels
        results = []
        for i, lbls in enumerate(labels, 1):
            try:
                results.append((i, davies_bouldin_score(data, lbls)))
            except ValueError:
                results.append((i, float('inf')))
        else:
            self.metrics = np.array(results)
            self.metrics[self.metrics[:, 1] == np.inf, 1] = self.metrics[self.metrics[:, 1] != np.inf, 1].max() * 1.1

        # Find the best votes
        with np.errstate(divide='ignore', invalid='ignore'):
            m = np.array(self.metrics)
            m[:, 1] = 1/m[:, 1]

        self.best_votes = self._get_best_votes(m)
        self.K = self.best_votes[0]
        return self.K


class WeightedEstimator:
    __estimators = dict(AsankaPerera=AsankaPerera, Riddle=Riddle, PhamDimovNguyen=PhamDimovNguyen,
                        GapStatistic=GapStatistic, Silhouette=Silhouette, CalinskiHarabasz=CalinskiHarabasz,
                        DaviesBouldin=DaviesBouldin)

    @check_types(estimators=(type(None), list, tuple))
    def __init__(self, estimators=None, verbose=0):
        self.estimators = self._check_estimators(estimators)
        self.scores = dict()
        self.verbose = verbose

    def _check_estimators(self, estimators):
        # If None is passed return all
        if not estimators:
            return [est() for est in self.__estimators.values()]

        # Otherwise check if all are valid
        _not_valid = [e for e in estimators if (e not in self.__estimators.keys()) &
                      (e.__class__ not in self.__estimators.values())]
        if _not_valid:
            raise KeyError(f'Some of the estimators chosen are not valid. '
                           f'Choose estimators from {self.__estimators.keys()}')

        # return instances of the estimators
        return [self.__estimators[e]() if isinstance(e, str) else e for e in estimators]

    @check_types(data=np.ndarray)
    def fit(self, data, inertias=None, labels=None, **kwargs):
        if not (inertias or labels):
            print('Estimators implemented need one of "inertias" or "labels" to be passed.')
            return

        _not_fitted = self.estimators.copy()
        for i, est in enumerate(self.estimators, 1):
            print_progress_bar(i, len(self.estimators), f'Fitting estimators ({est.__class__.__qualname__}): ',
                               level=1, verbose=self.verbose)
            try:
                est.fit_s_k(data=data, s_k=inertias, labels=labels, data_shape=data.shape, **kwargs)

            except Exception as e:
                print(f'{est.__class__.__name__} not fit due to exception.')
                printv(str(e), self.verbose, level=1)

            else:
                _not_fitted.remove(est)

        if _not_fitted == self.estimators:
            print('No estimator could be fitted.')

        return self

    @check_types(n_votes=int)
    def get_nr_cluster(self, n_votes=3):
        _est, n_votes = self._get_nr_cluster_sanity_checks(n_votes)
        _points = list(range(1, n_votes * 2, 2))[::-1]
        _max_score = _points[0] * len(_est)
        self.scores = Counter()

        for est in _est:
            for vote, pts in zip(est.best_votes[:n_votes], _points):
                if vote != 0:
                    self.scores[vote] += pts

        _high_score = max(self.scores.values())
        n = sorted([key for key, value in self.scores.items() if value == _high_score])

        printv(f'Best number of clusters is {n}, with a voting score of {_high_score*100/_max_score:.2f}%',
               self.verbose, level=1)
        return n[0]

    def _get_nr_cluster_sanity_checks(self, nr_votes):
        # filter estimators to include only fitted ones
        _not_fitted = [est.__class__.__name__ for est in self.estimators if len(est.best_votes) == 0]
        if _not_fitted:
            warnings.warn(f'Not all estimators are fitted. {_not_fitted} will be ignored. To include this '
                          f'estimators either pass them to the class already fitted or fit all of the estimators '
                          f'using the "fit" or "fit_s_k" methods of the class.', stacklevel=2)
        _est = [est for est in self.estimators if est.__class__.__name__ not in _not_fitted]

        # get max number of votes possible (min of _est)
        _nr_votes = min([len(est.best_votes) for est in _est])
        if _nr_votes < nr_votes:
            warnings.warn(f'Not all valid estimators have been fit for the number of votes requested. Using maximum '
                          f'nr_votes allowed: {_nr_votes}.', stacklevel=2)
            nr_votes = _nr_votes

        return _est, nr_votes

    def plot_metrics(self, n_votes=3):
        normalise = True  # todo add separate plots if normalise == False

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(24, 12))

        for est in self.estimators:
            est.plot_metric(ax, show=False, normalise=normalise, n_votes=n_votes)

        _verbose, self.verbose = self.verbose, 0
        k = self.get_nr_cluster(n_votes)
        self.verbose = _verbose

        ax.axvline(k, color='k', linestyle='--')
        ax.text(k, 1.01, f'Best K = {k}', rotation=45, size=10, transform=ax.get_xaxis_transform())

        fig.legend()
        fig.show()
