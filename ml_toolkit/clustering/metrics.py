from itertools import combinations
from joblib import Parallel, delayed

import numpy as np
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances

from ..utils.os_utl import filter_kwargs


# ############################################### FUNCTIONS #################################################
def cluster_density(x, labels, metric='euclidean', n_pts=-1):
    dens = list()
    max_per_label = list()
    for k in np.unique(labels):
        pts = x[labels == k]
        dist = pairwise_distances(pts, metric=metric)
        mean_dist = dist.mean(axis=1) if n_pts < 0 else np.sort(dist, axis=1)[:, 1:n_pts+1].mean(axis=1)
        dens.append(mean_dist.mean())
        max_per_label.append(mean_dist.max())

    dens = 1 - np.array(dens)/max(max_per_label)
    return np.unique(labels), np.round(dens, 3)


def local_cluster_density(x, labels, metric='euclidean', n_pts=10):
    return cluster_density(x, labels, metric, n_pts)


def gap_statistic(k, inertia, data_shape, n_refs=3, **kwargs):
    # Generate a reference inertia to compare to and
    # filter kwargs
    kwargs = filter_kwargs(kwargs, MiniBatchKMeans.__init__)
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = data_shape[0] // 100 if data_shape[0] > 100000 else 1000

    # For n references, generate random sample and perform clustering getting resulting dispersion of each loop
    ref_disps = np.zeros(n_refs)
    for i in range(n_refs):
        # Create new random reference set
        rand_ref = np.random.random_sample(size=data_shape)

        # Cluster this data and save the inertias
        ref_disps[i] = MiniBatchKMeans(k, **kwargs).fit(rand_ref).inertia_

    # return the log difference to the original inertia
    return np.log(np.mean(ref_disps)) - np.log(inertia)


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


# ######################################## METRICS CLASS ####################################################
class Metrics:
    __metrics = dict(GapStatistic=gap_statistic, Silhouette=silhouette_score_block, 
                     CalinskiHarabasz=calinski_harabasz_score, DaviesBouldin=davies_bouldin_score,
                     ClusterDensity=cluster_density, LocalClusterDensity=local_cluster_density)
    # todo inter cluster distance (distance between closest points in different clusters)

    def __init__(self, metrics=None):
        self._metrics = self._check_metrics(metrics)
        self.results = dict()

    def _check_metrics(self, metrics):
        # If None is passed return all
        if not metrics:
            return list(self.__metrics.values())

        # Otherwise check if all are valid
        _not_valid = [e for e in metrics if (e not in self.__metrics.keys()) &
                      (e not in self.__metrics.values())]
        if _not_valid:
            raise KeyError(f'Some of the metrics chosen are not valid. '
                           f'Choose metrics from {self.__metrics.keys()}')

        # return instances of the metrics
        return [self.__metrics[e] if isinstance(e, str) else e for e in metrics]

    def get_metrics(self, x, model, decimals=4, **kwargs):
        for func in self._metrics:
            _kwargs = filter_kwargs(kwargs, func)
            name = func.__name__

            if ('n_pts' in _kwargs) and (name == 'cluster_density'):
                _kwargs['n_pts'] = -1

            try:
                self.results[name] = np.round((func(x, model.labels_, **_kwargs) if func != gap_statistic else
                                               func(model.n_clusters, model.inertia_, x.shape, **_kwargs)),
                                              decimals)
            except Exception as e:
                print(f'Could not compute {name} due to exception: {e}.')

        return self.results
