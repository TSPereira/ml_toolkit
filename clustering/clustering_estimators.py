import math
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from itertools import combinations
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.externals.joblib import Parallel, delayed

from ..utils.os_utils import check_types


"""
ReciprocalDeltaLog, AsankaPerera and PhamDimovNguyen estimators are implemented according to João Paulo Figueira's 
post: https://towardsdatascience.com/what-is-k-ddf36926a752

GapStatistic estimator is the Miles Granger implementation 2016.04.25.1430 shared in 
https://anaconda.org/milesgranger/gap-statistic/notebook
"""


# TODO add silhouette, calinski harabasz, davies_bouldin
class Riddle:
    """
    Riddle K-estimator.
    Estimates the correct value for K using the reciprocal delta log rule.
    """

    def __init__(self, cluster_fn=MiniBatchKMeans):
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn

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

        diff = np.pad(np.diff(r_k), (1, 0), 'constant', constant_values=float('-inf'))
        f_diff = np.vstack((n_cl, diff/np.log(n_cl))).T

        self.best_votes = self._get_best_votes(f_diff)
        self.K = self.best_votes[0]

        return self.K

    @staticmethod
    def _get_best_votes(arr):
        k_best_n = arr[arr[:, 1].argsort()[::-1], 0].astype(int)
        return k_best_n.tolist()

    @staticmethod
    def _get_max_k(x, max_k):
        if max_k > x.shape[0]:
            print(f'Can only use number of clusters lower than number of examples ({x.shape[0]}).')
        return min(max_k, x.shape[0])


class AsankaPerera:
    """Estimates the K-Means K hyperparameter through geometrical analysis of the distortion curve"""

    def __init__(self, cluster_fn=MiniBatchKMeans, tolerance=1e-3):
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn
        self.tolerance = tolerance

    @check_types(x=np.ndarray, max_k=int)
    def fit(self, x, max_k=50, **kwargs):
        """
        Directly fit this estimator to original data considering max_k clusters.
        :param np.ndarray x: Array with the data to cluster
        :param int max_k: Maximum number of clusters to try to find
        :param kwargs: Tolerance to be used can be passed as a kwarg
        :return int: Best number of clusters
        """

        self._set_tolerance(**kwargs)
        max_k = self._get_max_k(x, max_k)

        s_k_list = list()
        sk0 = 0
        for k in range(1, max_k + 1):
            sk1 = self.cluster(k).fit(x).inertia_
            s_k_list.append(sk1)
            if k > 2 and abs(sk0 - sk1) < self.tolerance:
                break
            sk0 = sk1

        self._get_best_k(s_k_list)
        return self.K

    @check_types(s_k=(np.ndarray, list, tuple))
    def fit_s_k(self, s_k, **kwargs):
        """
        Fit the estimator to the distances of each datapoint to assigned cluster centroid
        :param np.ndarray|list|tuple s_k: Collection of inertias_ for each n_cluster explored
        :param kwargs: Tolerance to be used can be passed as a kwarg
        :return int: Best number of clusters
        """

        self._set_tolerance(**kwargs)
        if isinstance(s_k, list):
            s_k = np.array(s_k)

        s_k_list = list()
        sk0 = 0
        for k in range(len(s_k)):
            sk1 = s_k[k]
            s_k_list.append(sk1)
            if k > 2 and abs(sk0 - sk1) < self.tolerance:
                break
            sk0 = sk1

        self._get_best_k(s_k_list)
        return self.K

    @classmethod
    def find_largest_dist(cls, s_k, x0, y0, x1, y1):
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

    def _get_best_k(self, s_k_list):
        s_k = np.array(s_k_list)

        # Pass the line endpoints and find max dist
        k_dist = self.find_largest_dist(s_k, x0=1, y0=s_k[0], x1=3 * len(s_k), y1=0)
        self.best_votes = self._get_best_votes(k_dist)
        self.K = self.best_votes[0]
        return

    @staticmethod
    def _get_best_votes(arr):
        k_best_n = arr[arr[:, 1].argsort()[::-1], 0].astype(int)
        return k_best_n.tolist()

    def _set_tolerance(self, **kwargs):
        self.tolerance = kwargs['tolerance'] if 'tolerance' in kwargs else self.tolerance

    @staticmethod
    def _get_max_k(x, max_k):
        if max_k > x.shape[0]:
            print(f'Can only use number of clusters lower than number of examples ({x.shape[0]}).')
        return min(max_k, x.shape[0])


class PhamDimovNguyen:
    """Estimates the best value for K using Pham-Dimov-Nguyen method"""
    def __init__(self, cluster_fn=MiniBatchKMeans):
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn

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
        f_k = s_k[1:] / (a_k[1:] * s_k[:-1])
        f_k = np.vstack((range(1, len(f_k) + 1), f_k)).T

        # get K from where function result is minimum and if needed correct it
        self.best_votes = self._get_best_votes(f_k)
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
        k_best_n = np.pad(k_best_n, (0, 1), 'constant', constant_values=1)

        _pad_n = len(arr) - len(k_best_n)
        k_best_n = np.pad(k_best_n, (0, _pad_n), 'constant', constant_values=0)

        return list(k_best_n)

    @staticmethod
    def _get_max_k(x, max_k):
        if max_k > x.shape[0]:
            print(f'Can only use number of clusters lower than number of examples ({x.shape[0]}).')
        return min(max_k, x.shape[0])


class GapStatistic:
    """
    Class that implements the GapStatistic estimator as defined by Miles Granger (implementation 2016.04.25.1430)
    shared in  https://anaconda.org/milesgranger/gap-statistic/notebook
    """

    def __init__(self, n_refs=3, cluster_fn=MiniBatchKMeans):
        self.n_refs = n_refs
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn
        self.results = pd.DataFrame({'n_clusters': [], 'gap': []})

    def fit(self, x, max_k=50, **kwargs):
        """
        Directly fit this estimator to original data considering max_k clusters.
        :param np.ndarray x: Array with the data to cluster
        :param int max_k: Maximum number of clusters to try to find
        :param kwargs: additional arguments for cluster function passed, e.g: batch_size
        :return int: Best number of clusters
        """

        max_k = self._get_max_k(x, max_k)
        for gap_index, k in enumerate(range(1, max_k + 1)):
            ref_disp = self._random_samples(k, x.shape, **kwargs)

            # Get clusters for original data and respective dispersion
            if hasattr(self.cluster(), 'batch_size') & ('batch_size' not in kwargs):
                kwargs['batch_size'] = np.floor(x.shape[0] / 100).astype(int) if x.shape[0] > 100000 else 1000

            # fit data to cluster function provided
            orig_disp = self.cluster(k, **kwargs).fit(x).inertia_

            # Calculate gap statistic
            gap = np.log(ref_disp) - np.log(orig_disp)
            self.results = self.results.append({'n_clusters': k, 'gap': gap}, ignore_index=True)

        self.best_votes = self._get_best_votes(self.results.values)
        self.K = self.best_votes[0]

        return self.K

    def fit_s_k(self, s_k, data_shape, **kwargs):
        """
        Fit the estimator to the distances of each datapoint to assigned cluster centroid
        :param np.ndarray|list|tuple s_k: Collection of distances between each datapoint and its assigned cluster centroid
        :param np.ndarray|list|tuple data_shape: shape of the data used to cluster
        :param kwargs: additional arguments for cluster function passed, e.g: batch_size
        :return int: Best number of clusters
        """

        max_k = len(s_k)

        for gap_index, k in enumerate(range(1, max_k + 1)):
            ref_disp = self._random_samples(k, data_shape, **kwargs)
            orig_disp = s_k[gap_index]

            # Calculate gap statistic
            gap = np.log(ref_disp) - np.log(orig_disp)
            self.results = self.results.append({'n_clusters': k, 'gap': gap}, ignore_index=True)

        self.best_votes = self._get_best_votes(self.results.values)
        self.K = self.best_votes[0]

        return self.K

    def _random_samples(self, k, data_shape, **kwargs):
        # Holder for reference dispersion results
        ref_disps = np.zeros(self.n_refs)

        # For n references, generate random sample and perform clustering getting resulting dispersion of each loop
        for i in range(self.n_refs):
            # Create new random reference set
            rand_ref = np.random.random_sample(size=data_shape)

            if hasattr(self.cluster(), 'batch_size') & ('batch_size' not in kwargs):
                kwargs['batch_size'] = np.floor(rand_ref.shape[0] / 100).astype(int) if rand_ref.shape[0] > 100000 \
                    else 1000

            km = MiniBatchKMeans(k, **kwargs)
            km.fit(rand_ref)
            ref_disps[i] = km.inertia_

        return np.mean(ref_disps)

    @staticmethod
    def _get_best_votes(arr):
        k_best_n = arr[arr[:, 1].argsort()[::-1], 0].astype(int)
        return k_best_n.tolist()

    @staticmethod
    def _get_max_k(x, max_k):
        if max_k > x.shape[0]:
            print(f'Can only use number of clusters lower than number of examples ({x.shape[0]}).')
        return min(max_k, x.shape[0])


class Silhouette:
    def __init__(self):
        pass

    # def fit(self, x, max_k):

    # def fit_


class CalinskiHarabasz:
    def __init__(self):
        pass


class DaviesBouldin:
    def __init__(self):
        pass


class WeightedEstimator:
    __estimators = dict(AsankaPerera=AsankaPerera, Riddle=Riddle, PhamDimovNguyen=PhamDimovNguyen,
                        GapStatistic=GapStatistic)

    def __init__(self, estimators=('AsankaPerera', 'Riddle', 'PhamDimovNguyen', 'GapStatistic')):
        self.estimators = self._check_estimators(estimators)
        self.scores = dict()

    def _check_estimators(self, estimators):
        _not_valid = [e for e in estimators if (e not in self.__estimators.keys()) &
                      (e.__class__ not in self.__estimators.values())]

        if _not_valid:
            raise KeyError(f'Some of the estimators chosen are not valid. '
                           f'Choose estimators from {self.__estimators.keys()}')

        return [self.__estimators[e]() if isinstance(e, str) else e for e in estimators]

    def fit_s_k(self, s_k, data_shape, **kwargs):
        for est in self.estimators:
            print(est.__class__.__name__)  # todo add progress bar
            _kwargs = {key: value for key, value in kwargs.items() if hasattr(est.cluster(), key) or hasattr(est, key)}

            est.fit_s_k(s_k, data_shape=data_shape, **_kwargs)

    def fit(self, x, max_k=50, **kwargs):
        for est in self.estimators:  # todo add progress bar
            _kwargs = {key: value for key, value in kwargs.items() if hasattr(est.cluster(), key) or hasattr(est, key)}

            est.fit(x, max_k, **_kwargs)

    def get_nr_cluster(self, nr_votes=3):
        _est, nr_votes = self._get_nr_cluster_sanity_checks(nr_votes)
        _points = list(range(1, nr_votes*2, 2))[::-1]
        _max_score = _points[0] * len(_est)
        self.scores = dict()

        for est in _est:
            i = 0
            for vote in est.best_votes[:nr_votes]:
                if vote != 0:
                    if str(vote) in self.scores:
                        self.scores[str(vote)] += _points[i]
                    else:
                        self.scores[str(vote)] = _points[i]
                    i += 1

        _high_score = max(self.scores.values())
        keys = [key for key, value in self.scores.items() if value == _high_score]
        print(f'Best number of clusters is {keys[0] if len(keys) == 1 else keys}, with a voting score of '
              f'{_high_score*100/_max_score}%')

        return keys

    def _get_nr_cluster_sanity_checks(self, nr_votes):
        _votes_len = {est.__class__.__name__: len(est.best_votes) for est in self.estimators}
        _not_fitted = [key for key, value in _votes_len.items() if value == 0]

        if _not_fitted:
            print(f'Not all estimators are fitted. {_not_fitted} will be ignored. To include this estimators either '
                  f'pass them to the class already fitted or fit all of the estimators using the "fit" or "fit_s_k" '
                  f'methods of the class.')

        _est = [est for est in self.estimators if est.__class__.__name__ not in _not_fitted]

        # get max number of votes possible (min of _est)
        _nr_votes = min([len(est.best_votes) for est in _est])

        if _nr_votes < nr_votes:
            print(f'Not all valid estimators have been fit for the number of votes requested. Using maximum nr_votes '
                  f'allowed: {_nr_votes}.')
            nr_votes = _nr_votes

        return _est, nr_votes


# ############################################### FUNCTIONS #################################################
def silhouette_score_block(x, labels, metric='euclidean', sample_size=None,
                           random_state=None, n_jobs=1, **kwargs):
    """Compute the mean Silhouette Coefficient of all samples.
    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``. To clarify,
    b is the distance between a sample and the nearest cluster that b is not a part of. This function returns the
    mean Silhouette Coefficient over all samples. To obtain the values for each sample, use silhouette_samples .The
    best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally
    indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.

    :param numpy.ndarray x: feature array of shape (n_samples_a, n_features)
    :param numpy.ndarray labels: label values for each sample as an array of shape (n_samples, )
    :param string metric: default: 'euclidean'. The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options allowed by
        metrics.pairwise.pairwise_distances. If X is the distance array itself, use "precomputed" as the metric.
    :param int sample_size: The size of the sample to use when computing the Silhouette Coefficient. If sample_size
    is None, no sampling is used.
    :param int|numpy.RandomState random_state: Optional. The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random number generator.
    :param int n_jobs: number of processing cores to use. -1 for all
    :param kwargs: optional keyword parameters. Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still metric dependent. See the scipy docs for
        usage examples.

    :return float: silhouette. Mean Silhouette Coefficient for all samples.


    References
    ----------
    Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
        Interpretation and Validation of Cluster Analysis". Computational
        and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
    http://en.wikipedia.org/wiki/Silhouette_(clustering)
    """

    if sample_size is not None:
        random_state = check_random_state(random_state)
        indices = random_state.permutation(x.shape[0])[:sample_size]
        if metric == "precomputed":
            raise ValueError('Distance matrix cannot be precomputed')
        else:
            x, labels = x[indices], labels[indices]
    return np.mean(silhouette_samples_block(
        x, labels, metric=metric, n_jobs=n_jobs, **kwargs))


def silhouette_samples_block(x, labels, metric='euclidean', n_jobs=1, **kwargs):
    """Compute the Silhouette Coefficient for each sample.

    :param numpy.ndarray x: feature array of shape (n_samples_a, n_features)
    :param numpy.ndarray labels: label values for each sample as an array of shape (n_samples, )
    :param string metric: default: 'euclidean'. The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options allowed by
        metrics.pairwise.pairwise_distances. If X is the distance array itself, use "precomputed" as the metric.
    :param int n_jobs: number of processing cores to use. -1 for all
    :param kwargs: optional keyword parameters. Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still metric dependent. See the scipy docs for
        usage examples.

    :return numpy.ndarray: silhouette values. Clusters of size 1 have silhouette 0

    """

    a = _intra_cluster_distances_block(x, labels, metric, n_jobs=n_jobs,
                                       **kwargs)
    b = _nearest_cluster_distance_block(x, labels, metric, n_jobs=n_jobs,
                                        **kwargs)
    sil_samples = (b - a) / np.maximum(a, b)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def _intra_cluster_distances_block_(sub_x, metric, **kwargs):
    """Calculates the intra cluster distances for each cluster

    :param numpy.ndarray sub_x: subset of all the samples that have the same cluster. array of shape
    (n_samples, n_features)
    :param string metric: The metric to use when calculating distance between instances in a feature array. If metric
        is a string, it must be one of the options allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    :param kwargs: optional keyword parameters. Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still metric dependent. See the scipy docs for
        usage examples.
    :return float: intra_cluster mean pairwise distance value
    """

    distances = pairwise_distances(sub_x, metric=metric, **kwargs)
    return distances.sum(axis=1) / (distances.shape[0] - 1)


# noinspection PyUnresolvedReferences
def _intra_cluster_distances_block(x, labels, metric, n_jobs=1, **kwargs):
    """Calculate the mean intra-cluster distance for sample i.

    :param numpy.ndarray x: feature array of shape (n_samples_a, n_features)
    :param numpy.ndarray labels: label values for each sample as an array of shape (n_samples, )
    :param string metric: default: 'euclidean'. The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options allowed by
        metrics.pairwise.pairwise_distances. If X is the distance array itself, use "precomputed" as the metric.
    :param int n_jobs: number of processing cores to use. -1 for all
    :param kwargs: optional keyword parameters. Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still metric dependent. See the scipy docs for
        usage examples.

    :return numpy.ndarray: shape (n_samples). Mean intra-cluster distance
    """

    intra_dist = np.zeros(labels.size, dtype=float)
    values = Parallel(n_jobs=n_jobs)(delayed(_intra_cluster_distances_block_)(
        x[np.where(labels == label)[0]], metric, **kwargs) for label in np.unique(labels))
    for label, values_ in zip(np.unique(labels), values):
        intra_dist[np.where(labels == label)[0]] = values_
    return intra_dist


def _nearest_cluster_distance_block_(sub_x_a, sub_x_b, metric, **kwargs):
    """Calculate the mean nearest-cluster distance for sample i.

    :param numpy.ndarray sub_x_a: subset of all the samples that have the same cluster. array of shape
    (n_samples, n_features)
    :param numpy.ndarray sub_x_b: subset of all the samples that have the same cluster (different from the cluster
    represented by sub_x_a). array of shape (n_samples, n_features)
    :param string metric: The metric to use when calculating distance between instances in a feature array. If metric
        is a string, it must be one of the options allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    :param kwargs: optional keyword parameters. Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still metric dependent. See the scipy docs for
        usage examples.
    :return float: intra_cluster mean pairwise distance value
    """

    dist = pairwise_distances(sub_x_a, sub_x_b, metric=metric, **kwargs)
    dist_a = dist.mean(axis=1)
    dist_b = dist.mean(axis=0)
    return dist_a, dist_b


def _nearest_cluster_distance_block(x, labels, metric, n_jobs=1, **kwargs):
    """Calculate the mean nearest-cluster distance for sample i.

    :param numpy.ndarray x: feature array of shape (n_samples_a, n_features)
    :param numpy.ndarray labels: label values for each sample as an array of shape (n_samples, )
    :param string metric: default: 'euclidean'. The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options allowed by
        metrics.pairwise.pairwise_distances. If X is the distance array itself, use "precomputed" as the metric.
    :param int n_jobs: number of processing cores to use. -1 for all
    :param kwargs: optional keyword parameters. Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still metric dependent. See the scipy docs for
        usage examples.

    :return numpy.ndarray: shape (n_samples). Mean intra-cluster distance
    """

    # noinspection PyUnresolvedReferences
    inter_dist = np.empty(labels.size, dtype=float)
    inter_dist.fill(np.inf)
    # Compute cluster distance between pairs of clusters
    unique_labels = np.unique(labels)

    values = Parallel(n_jobs=n_jobs)(
        delayed(_nearest_cluster_distance_block_)(
            x[np.where(labels == label_a)[0]], x[np.where(labels == label_b)[0]], metric, **kwargs)
        for label_a, label_b in combinations(unique_labels, 2))

    for (label_a, label_b), (values_a, values_b) in zip(combinations(unique_labels, 2), values):
        indices_a = np.where(labels == label_a)[0]
        inter_dist[indices_a] = np.minimum(values_a, inter_dist[indices_a])
        del indices_a
        indices_b = np.where(labels == label_b)[0]
        inter_dist[indices_b] = np.minimum(values_b, inter_dist[indices_b])
        del indices_b
    return inter_dist


if __name__ == '__main__':
    pass

    print(1)