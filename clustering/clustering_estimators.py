import math
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from .silhouette_custom import silhouette_samples_block, silhouette_score_block


"""
ReciprocalDeltaLog, AsankaPerera and PhamDimovNguyen estimators are implemented according to João Paulo Figueira's 
post: https://towardsdatascience.com/what-is-k-ddf36926a752

GapStatistic estimator is the Miles Granger implementation 2016.04.25.1430 shared in 
https://anaconda.org/milesgranger/gap-statistic/notebook
"""


# TODO add silhouette, calinski harabasz, davies_bouldin
class Riddle(object):
    """
    Riddle K-estimator.
    Estimates the correct value for K using the reciprocal delta log rule.
    """


    def __init__(self, cluster_fn=MiniBatchKMeans):
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn

    def fit(self, x, max_k=50, **kwargs):
        # calculate s_k
        s_k = np.array([self.cluster(x, k) for k in range(1, max_k + 1)])
        return self.fit_s_k(s_k)

    def fit_s_k(self, s_k, **kwargs):
        if isinstance(s_k, list):
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


class AsankaPerera(object):
    """Estimates the K-Means K hyperparameter through geometrical analysis of the distortion curve"""

    def __init__(self, cluster_fn=MiniBatchKMeans, tolerance=1e-3):
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn
        self.tolerance = tolerance

    @staticmethod
    def distance_to_line(x0, y0, x1, y1, x2, y2):
        """
        Calculates the distance from (x0,y0) to the
        line defined by (x1,y1) and (x2,y2)
        """
        dx = x2 - x1
        dy = y2 - y1
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / \
               math.sqrt(dx * dx + dy * dy)

    def fit(self, x, max_k=50, **kwargs):
        """Fits the value of K"""

        self._set_tolerance(**kwargs)
        self.K = 0
        s_k_list = list()
        sk0 = 0

        for k in range(1, max_k + 1):
            sk1 = self.cluster(x, k)
            s_k_list.append(sk1)
            if k > 2 and abs(sk0 - sk1) < self.tolerance:
                break
            sk0 = sk1

        s_k = np.array(s_k_list)

        # Pass the line endpoints and find max dist
        k_dist = self.find_largest_dist(s_k, x0=1, y0=s_k[0], x1=3*len(s_k), y1=0)
        self.best_votes = self._get_best_votes(k_dist)
        self.K = self.best_votes[0]
        return self.K

    def fit_s_k(self, s_k, **kwargs):
        """Fits the value of K using the s_k series

        :param list|numpy.ndarray s_k:
        :param int nr_samples:
        :param float tolerance:
        :return:
        """

        self._set_tolerance(**kwargs)
        self.K = 0

        if isinstance(s_k, list):
            s_k = np.array(s_k)

        s_k_list = list()
        sk0 = 0

        # Fit the maximum K
        for k in range(len(s_k)):
            sk1 = s_k[k]
            s_k_list.append(sk1)
            if k > 2 and abs(sk0 - sk1) < self.tolerance:
                break
            sk0 = sk1

        s_k = np.array(s_k_list)

        # Pass the line endpoints and find max dist
        k_dist = self.find_largest_dist(s_k, x0=1, y0=s_k[0], x1=3*len(s_k), y1=0)
        self.best_votes = self._get_best_votes(k_dist)
        self.K = self.best_votes[0]

        return self.K

    def find_largest_dist(self, s_k, x0, y0, x1, y1):
        k_dist = np.array([[k, self.distance_to_line(k, s_k[k - 1], x0, y0, x1, y1)] for k in range(1, len(s_k) + 1)])
        if int(k_dist[k_dist[:, 1].argmax(), 0]) == int(k_dist[-1, 0]):
            print('AsankaPerera: Number of clusters explored is not optimal! Run analysis with higher number of '
                  'clusters.')
        return k_dist

    @staticmethod
    def _get_best_votes(arr):
        k_best_n = arr[arr[:, 1].argsort()[::-1], 0].astype(int)
        return k_best_n.tolist()

    def _set_tolerance(self, **kwargs):
        self.tolerance = kwargs['tolerance'] if 'tolerance' in kwargs else self.tolerance


class PhamDimovNguyen:
    """Estimates the best value for K using Pham-Dimov-Nguyen method"""
    def __init__(self, cluster_fn=MiniBatchKMeans):
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn

    @staticmethod
    def alpha_k(a_k, k, dim):
        if k == 2:
            ak = 1.0 - 3.0 / (4.0 * dim)
        else:
            ak1 = a_k[k - 1]
            ak = ak1 + (1.0 - ak1) / 6.0
        return ak

    def fit(self, x, max_k=50, **kwargs):
        # calculate s_k
        s_k = np.array([self.cluster(x, k) for k in range(1, max_k + 1)])
        return self.fit_s_k(s_k, data_shape=x.shape)

    def fit_s_k(self, s_k, data_shape, **kwargs):
        """Fits the value of K usinnr_featuresg the s_k series"""
        if isinstance(s_k, list):
            s_k = np.array(s_k)

        # calculate all alphas
        a_k = np.zeros(len(s_k) + 1)
        for i in range(2, len(s_k) + 1):
            a_k[i] = self.alpha_k(a_k, i, data_shape[1])

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
    def _get_best_votes(arr):
        """

        :param numpy.ndarray arr:
        :param n:
        :return list:
        """

        _arr = arr[arr[:, 1] <= 0.85]
        k_best_n = _arr[_arr[:, 1].argsort(), 0].astype(int)
        k_best_n = np.pad(k_best_n, (0, 1), 'constant', constant_values=1)

        _pad_n = len(arr) - len(k_best_n)
        k_best_n = np.pad(k_best_n, (0, _pad_n), 'constant', constant_values=0)

        return k_best_n.tolist()


class GapStatistic(object):
    def __init__(self, n_refs=3, cluster_fn=MiniBatchKMeans):
        self.n_refs = n_refs
        self.K = 0
        self.best_votes = list()
        self.cluster = cluster_fn
        self.results = pd.DataFrame({'n_clusters': [], 'gap': []})

    def fit(self, x, max_k=50, **kwargs):
        for gap_index, k in enumerate(range(1, max_k + 1)):
            ref_disp = self._random_samples(k, x.shape, **kwargs)

            # Get clusters for original data and respective dispersion
            if hasattr(self.cluster(), 'batch_size') & ('batch_size' not in kwargs):
                kwargs['batch_size'] = np.floor(x.shape[0] / 100).astype(int) if x.shape[0] > 100000 else 1000

            km = self.cluster(k, **kwargs)
            km.fit(x)
            orig_disp = km.inertia_

            # Calculate gap statistic
            gap = np.log(ref_disp) - np.log(orig_disp)
            self.results = self.results.append({'n_clusters': k, 'gap': gap}, ignore_index=True)

        self.best_votes = self._get_best_votes(self.results.values)
        self.K = self.best_votes[0]

        return self.K

    def fit_s_k(self, s_k, data_shape, **kwargs):
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


class Silhouette(object):
    def __init__(self):
        pass

    # def fit(self, x):


class CalinskiHarabasz(object):
    def __init__(self):
        pass


class DaviesBouldin(object):
    def __init__(self):
        pass


class WeightedEstimator(object):
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

