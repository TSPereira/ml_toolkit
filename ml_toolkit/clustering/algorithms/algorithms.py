from copy import deepcopy
from inspect import isclass
from itertools import cycle
from operator import attrgetter
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.exceptions import NotFittedError
from scipy.spatial import ConvexHull

from .snn import SNN
from .. import WeightedEstimator, Metrics
from ...utils.os_utl import filter_kwargs, check_types
from ...utils.log_utl import printv, print_progress_bar


_methods = dict(kmeans=KMeans, minibatchkmeans=MiniBatchKMeans, dbscan=DBSCAN, optics=OPTICS, snn=SNN)
try:
    from hdbscan import HDBSCAN
    _methods['hdbscan'] = HDBSCAN
except ModuleNotFoundError:
    warnings.warn('\nHDBSCAN could not be imported. To add it to the options install it in your environment.',
                  stacklevel=2)


class AgglomerativeClustering(AgglomerativeClustering):
    # todo
    #  compute_inertias
    #  add n_clusters
    def _compute_linkage_matrix(self):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(self.children_.shape[0])
        n_samples = len(self.labels_)
        for i, merge in enumerate(self.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        return np.column_stack([self.children_, self.distances_, counts]).astype(float)

    def plot_dendrogram(self, truncate_mode='level', p=3, **kwargs):
        from scipy.cluster.hierarchy import dendrogram

        linkage_matrix = self._compute_linkage_matrix()

        # Plot the corresponding dendrogram
        plt.title('Hierarchical Clustering Dendrogram')
        dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p, **kwargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()


class Cluster:
    __methods = _methods
    __predictor = KNeighborsClassifier

    @check_types(estimate_best_k=bool, max_k=int, verbose=int)
    def __init__(self, method='minibatchkmeans', predictor=None, estimate_best_k=False, max_k=50, metrics=None,
                 verbose=0, **kwargs):
        # kwargs can be passed as method__argname or predictor__argname
        self.c_model, self.p_model = self._validate_args(method, predictor, kwargs)
        self.est = None
        self.max_k = max_k
        self.estimate_best_k = estimate_best_k
        self.metrics = Metrics(metrics)
        self.verbose = verbose

    def _validate_args(self, method, predictor, kwargs):
        # split kwargs
        m_kwargs = {key[7:]: value for key, value in kwargs.items() if key.startswith('method_')}
        p_kwargs = {key[10:]: value for key, value in kwargs.items() if key.startswith('predictor_')}
        self._e_kwargs = {key[10:]: value for key, value in kwargs.items() if key.startswith('estimator_')}
        self._e_kwargs = filter_kwargs(self._e_kwargs, WeightedEstimator.__init__)

        # validate method
        if isinstance(method, tuple(self.__methods.values())):
            method = method
        elif (isinstance(method, str)) and (method in self.__methods.keys()):
            m_kwargs = filter_kwargs(m_kwargs, self.__methods[method].__init__)
            method = self.__methods[method](**m_kwargs)
        elif method in self.__methods.values():
            m_kwargs = filter_kwargs(m_kwargs, method.__init__)
            method = method(**m_kwargs)
        else:
            raise KeyError(f'Method passed is not valid. Chose one of {list(self.__methods.keys())} or pass an '
                           f'instance of one of {list(self.__methods.values())}')

        # validate predictor
        if predictor is None:
            if hasattr(method, 'predict'):
                predictor = method

            else:
                p_kwargs = filter_kwargs(p_kwargs, self.__predictor.__init__)
                predictor = self.__predictor(**p_kwargs)

        elif isclass(predictor):
            p_kwargs = filter_kwargs(p_kwargs, predictor.__init__)
            predictor = predictor(**p_kwargs)

        if not (hasattr(predictor, 'fit') and hasattr(predictor, 'predict')):
            raise NotImplementedError(f'"predictor" passed does not have both "fit" and "predict" methods.')

        return method, predictor

    def _fit_predictor(self, x, y, **kwargs):
        x_train, x_test, y_train, y_test = train_test_split(x, y, **filter_kwargs(kwargs, train_test_split))
        self.p_model.fit(x_train, y_train)

        # eval
        yhat_train, yhat = self.p_model.predict(x_train), self.p_model.predict(x_test)
        self._predictor_metrics = dict(train=dict(classification_report=classification_report(y_train, yhat_train),
                                                  confusion_matrix=confusion_matrix(y_train, yhat_train)),
                                       test=dict(classification_report=classification_report(y_test, yhat),
                                                 confusion_matrix=confusion_matrix(y_test, yhat)))

    def find_best_k(self, x, n_votes=3):
        mdls = dict()
        for i in range(1, self.max_k + 1):
            mdls[i] = mdl = deepcopy(self.c_model)
            mdl.n_clusters = i
            mdl.fit(x)
            print_progress_bar(i, self.max_k, prefix='Searching best k: ', level=1, verbose=self.verbose)

        inertias = list(map(attrgetter('inertia_'), mdls.values()))
        labels = list(map(attrgetter('labels_'), mdls.values()))

        self.est = WeightedEstimator(verbose=self.verbose, **self._e_kwargs).fit(x, inertias, labels)
        return self.est.get_nr_cluster(n_votes)

    def fit(self, x, **kwargs):
        if self.estimate_best_k:
            if not hasattr(self.c_model, 'n_clusters'):
                warnings.warn(f'\n{self.c_model.__class__.__qualname__} does not have "n_clusters" attribute. '
                              f'Only clustering methods with this attribute can use the "estimate_best_k" option.'
                              f'\nSkipping.', stacklevel=2)

            else:
                # todo raise some warning about time?
                self.c_model.n_clusters = self.find_best_k(x, n_votes=kwargs.get('est_n_votes', 3))

        # fit the clustering model
        printv('Fitting the clusterer.', self.verbose, level=1)
        self.c_model.fit(x)

        # Prepare a predictor
        # if the p_model is already fitted, it means p_model == c_model, else fit the predictor
        try:
            check_is_fitted(self.p_model)

        except NotFittedError:
            printv('Fitting the predictor.', self.verbose, level=1)
            self._fit_predictor(x, self.c_model.labels_, test_size=0.2, shuffle=True, stratify=self.c_model.labels_)

        return self

    def predict(self, x):
        return self.p_model.predict(x)

    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)

    def get_metrics(self, x, **kwargs):
        return self.metrics.get_metrics(x, self.c_model, **kwargs)

    def plot(self, x, labels=None, density=True, density_metric='euclidean', show=True, s=15, fontsize=12,
             show_areas=False, figsize=(30, 15)):
        if x.shape[1] > 2:
            raise NotImplementedError('plot method is not prepared to more than 2 dimensions.')

        # select labels
        labels = labels or self.c_model.labels_

        # Initiate figure
        fig, ax = plt.subplots(figsize=figsize)

        # compute density if needed
        dens = []
        if density:
            from ..metrics import cluster_density
            _, dens = cluster_density(x, labels, density_metric)

        # plot clusters
        for k, color in zip(np.unique(labels), cycle(TABLEAU_COLORS)):
            vals = x[labels == k]
            ax.scatter(*vals.T, label=k, s=s, color=color)

            if show_areas:
                hull = ConvexHull(vals)
                ax.fill(vals[hull.vertices, 0], vals[hull.vertices, 1], color, alpha=0.1)
                ax.plot(vals[hull.vertices, 0], vals[hull.vertices, 1], 'k', lw=0.5)

            text = k if (not density) else f'{k} ({dens[k]})'
            ax.text(*vals.mean(axis=0), s=text, fontsize=fontsize, weight='bold',
                    bbox=dict(facecolor='white', edgecolor='k', alpha=0.7))

        # plot title
        ax.set_title('Segment representation')

        if show:
            fig.show()

        return fig
