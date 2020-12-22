import numpy as np
import scipy as sp
import scipy.sparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_clusters(x, labels, cluster_centers=None, n_components=2, perplexity=100, verbose=False, random_state=0):
    """
    Function to plot representation of obtained clusters. Both representations in 2D or 3D can be produced

    :param x:               encoded data used for clustering
    :param labels:          labels of each cluster for each record in x
    :param cluster_centers: records of the representative centroid of each cluster (output of kmeans)
    :param n_components:    Dimension to plot on. 2 or 3 are the only ones supported
    :param perplexity:      Parameter of the TSNE embedding
    :param verbose:         Parameter to control the info displayed to the console
    :param random_state:    Parameter to guarantee reproducibility
    :return: fig            Returns the figure generated. If interactive mode is on figure will also be shown as pop-up
    """

    print('Plotting cluster {}D visualization...'.format(n_components))

    # Find number of clusters
    n_clusters = len(np.unique(labels))

    # If info about the center of the clusters is fed attach their position so it goes over PCA and TSNE
    if cluster_centers is not None:
        x = sp.sparse.vstack([x, cluster_centers])

    x = sp.sparse.csr_matrix.todense(x)
    # If there are more than 50 features, perform PCA to reduce them
    if x.shape[1] > 50:
        if verbose:
            print('Performing PCA...')
        pca = PCA(n_components=50, random_state=random_state)
        x_pca = pca.fit_transform(x)
    else:
        x_pca = x

    # Perform TSNE to reduce the features to 2D or 3D
    if verbose:
        print('Performing TSNE...')
    x_embedded = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(x_pca)

    # If info about the center of the clusters is fed separate our vectors from the centers
    centers = None
    if cluster_centers is not None:
        points = x_embedded[:-n_clusters, :]
        centers = x_embedded[-n_clusters:, :]
    else:
        points = x_embedded

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.suptitle('Perplexity: {}'.format(perplexity))
    cmap = plt.get_cmap('gist_rainbow')

    # For 3D
    if n_components == 3:
        ax.remove()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=5, cmap=cmap)

        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=200, c=np.unique(labels), marker='+', cmap=cmap)

    # For 2D
    elif n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, s=10, cmap=cmap)

        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], s=200, c=np.unique(labels), marker='+', cmap=cmap)

    else:
        print('Can\'t represent the number of dimensions requested.\n')
        return

    fig.colorbar(scatter)
    # plt.show(block=False)

    return fig


def plot_metrics(scores, key, metrics=None, method=None, save_path=None, name=''):
    """
    Function to plot the evaluated metrics during the clustering process. It can plot for a 1D key or 2D key
    :param scores: array with the metrics to plot
    :param key: axis values to plot the metrics on
    :param metrics: metrics used in the clustering process. These metrics will be used to determine the titles in the plots
    and should come in the same order as when passed to the clustering class
    :param method: method used for clustering. depending on the method, the format of the array and key might be different
    for example:
        'kmeans': scores = [n_clusters.shape[0], ['sum of squared errors', metrics]]
                  key = ['n_clusters']
        'DBSCAN': scores = [eps.shape[0], min_samples.shape[0], ['n_clusters_estimated', '% mislabeled records', metrics]]
                  key = ['eps', 'min_samples']
    :param save_path: path to save the plots to. If no path is given, the figure will be shown on the screen
    :param name: name to use for the plot (before the metric title)
    :return:
    """
    fig = False

    # Define titles
    if (metrics is not None) & (method is not None):
        titles = _get_titles_from_metrics(metrics, method)
    else:
        titles = None

    # For DBSCAN hyperparameters (two)
    if len(scores.shape) == 3:
        fig = plot_mesh3d(key[0], key[1], scores, titles=titles)

    # For kmeans nr_clusters hyperparameter
    if len(scores.shape) == 1 or len(scores.shape) == 2:
        fig = plot_mesh2d(key, scores, titles=titles)

    # Save or show plot
    if fig:
        name = '{} evaluation metrics'.format(name)
        fig.suptitle(name)
        if save_path:
            fig.savefig(save_path + name.replace(' ', '_'))
        else:
            fig.show()

    return


def plot_mesh3d(x, y, z, axes_labels=None, titles=None):
    """
    Function to plot 3D surfaces
    If z array is 3D then it will plot multiple plots in the same figure. If only 2D then a single plot will be made.

    :param x: x-axis array
    :param y: y-axis array
    :param z: z-axis array
    :param axes_labels: names to give to the axes
    :param titles: name to give to each plot
    :return:
    """
    fig = plt.figure(figsize=(20, 10))
    # Find how many plots to do and prepare data for them
    if len(z.shape) == 3:
        # Multiplot
        nr_plots = z.shape[2]
        Z = [z[:, :, i] for i in range(z.shape[2])]
    elif len(z.shape) == 2:
        nr_plots = 1
        Z = z
    else:
        print('Wrong number of dimensions to plot')
        return

    # Check how to setup window
    if np.sqrt(nr_plots).is_integer():
        rows = cols = int(np.sqrt(nr_plots))
    else:
        cols = int(np.ceil(np.sqrt(nr_plots)))
        rows = int(nr_plots//cols + (1 if nr_plots % cols > 0 else 0))

    # Plots each plot in turn
    for i in range(nr_plots):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')

        X, Y = np.meshgrid(x, y)
        ax.scatter(X, Y, Z[i], marker='.')

        if axes_labels:
            try:
                ax.set_xlabel(axes_labels[0])
                ax.set_ylabel(axes_labels[1])
                ax.set_zlabel(axes_labels[2+i])
            except:
                print('Could not set axes labels as the vector inserted is not of the correct dimension')
                pass

        if titles:
            try:
                ax.set_title(titles[i])
            except:
                print('Could not set title as the vector inserted is not of the correct dimension')
                pass

    # plt.show(block=False)

    return fig


def plot_mesh2d(x, y, axes_labels=None, titles=None):
    """
    Function to plot 2D lines
    If y array is 2D then it will plot multiple plots in the same figure. If only 1D then a single plot will be made.

    :param x: x-axis array
    :param y: y-axis array
    :param axes_labels: names to give to the axes
    :param titles: name to give to each plot
    :return:
    """

    fig = plt.figure(figsize=(20, 10))
    # Find how many plots to do and prepare data for them
    if len(y.shape) == 2:
        # Multiplot
        nr_plots = y.shape[1]
    elif len(y.shape) == 1:
        nr_plots = 1
    else:
        print('Wrong number of dimensions to plot')
        return

    # Check how to setup window
    if np.sqrt(nr_plots).is_integer():
        rows = cols = int(np.sqrt(nr_plots))
    else:
        cols = int(np.ceil(np.sqrt(nr_plots)))
        rows = int(nr_plots // cols + (1 if nr_plots % cols > 0 else 0))

    # Plots each plot in turn
    for i in range(nr_plots):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.plot(x, y[:, i], marker='.')

        if axes_labels:
            try:
                ax.set_xlabel(axes_labels[0])
                ax.set_ylabel(axes_labels[1+i])
            except:
                print('Could not set axes labels as the vector inserted is not of the correct dimension')
                pass

        if titles:
            try:
                ax.set_title(titles[i])
            except:
                print('Could not set title as the vector inserted is not of the correct dimension')
                pass

    # plt.show(block=False)

    return fig


def take_outliers_graph(val, cutoff=0.01):
    """
    Function to remove outliers for graph plot

    :param val: list, array
    :param cutoff: value for number of records to delete on each side (percentage between 0 and 1. Std = 0.01)
    :return: val: list, array reduced
    """

    if not (0 < cutoff < 1):
        print('Value given for outliers cutoff is not between 0 and 1. Using 0.01 as std cutoff')
        cutoff = 0.01

    # Get the histogram bins and calculate the frequency in each
    hst = np.histogram(val, bins=100)
    per = hst[0] / np.sum(hst[0])

    # find the upper limit
    acc = 0
    i = len(per)
    while acc < cutoff:
        i -= 1
        acc += per[i]
    val_above = hst[1][i + 1]
    val = val[val <= val_above]

    # find the lower limit
    acc = 0
    i = -1
    while acc < cutoff:
        i += 1
        acc += per[i]

    if i > 0:
        val_below = hst[1][i - 1]
        val = val[val >= val_below]
    return val


def _get_titles_from_metrics(metrics, method):
    """
    Function to define the plots titles based on the method and metrics used in the clustering
    :param metrics: list with the metrics used (ordered in the same way as given to the clustering algorithm)
    :param method: string with the method used for clustering. if method is not supported then it will not return any title
    :return:
    """

    if method in ['kmeans', 'minibatchkmeans', 'kmodes', 'kmedoids', 'kprototypes']:
        titles = ['Sum of distances'] + metrics
    elif method == 'DBSCAN':
        titles = ['Nr clusters estimated', 'Percentage of non-classified'] + metrics
    else:
        titles = None

    return titles
