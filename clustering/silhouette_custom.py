import numpy as np
from itertools import combinations
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.externals.joblib import Parallel, delayed


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
