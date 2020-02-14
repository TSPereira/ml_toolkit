import numpy as np
from numpy.linalg import inv, pinv


def _covariance_to_correlation_matrix(covariance_matrix):
    D = np.sqrt(np.diag(covariance_matrix))
    DInv = inv(D).r
    correlation_matrix = DInv * covariance_matrix * DInv
    return correlation_matrix


def _is_symmetric_matrix(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)


def _drop_columns_with_zeros(x_array):
    return x_array[:, ~np.all(x_array == 0, axis=0)]


def _drop_equal_columns_and_rows(matrix, index=None, return_index=False):
    if not _is_symmetric_matrix(matrix):
        raise AssertionError('The matrix is not symmetric.')

    if index is None:
        _, index = np.unique(matrix, axis=0, return_index=True)

    if return_index:
        return matrix[np.sort(index)].T[np.sort(index)], np.sort(index)
    else:
        return matrix[np.sort(index)].T[np.sort(index)]


def _drop_linear_dependent_columns_and_rows(matrix, index=None, return_index=False):
    if not _is_symmetric_matrix(matrix):
        raise AssertionError('The matrix is not symmetric.')

    if index is None:
        # _, index = Calculated_Matrixes(matrix).rref()
        lambdas, _ = np.linalg.eig(matrix.T)

        index = lambdas == 0
    if return_index:
        return matrix[index].T[index], index
    else:
        return matrix[index].T[index]


def scatter_separability_score(x_array, labels_vector, drop_linear_dependent=True):
    """
    Calculate the scatter separability score to access separation of clusters
    :param x_array: numpy array of the subset of features used
    :param labels_vector: labels vector to calculate the separation on
    :param drop_linear_dependent: whether to drop features that are linear dependant or not
    :return: return score
    """

    # Remove unnecessary columns to obtain relevant computations
    x_array = _drop_columns_with_zeros(x_array)
    n_features = x_array.shape[1]

    # Get the probability of an element belonging to a cluster
    labels_unique, labels_count = np.unique(labels_vector, return_counts=True)
    probabilities = labels_count/sum(labels_count)

    means = []
    covariances = []

    for label in labels_unique:
        x_k = x_array[labels_vector == label, :]

        means.append(np.average(x_k, axis=0))
        covariances.append(np.cov(x_k.transpose()))

    m0 = 0

    for label in range(len(labels_unique)):

        m0 += probabilities[label] * means[label]

    s_within = np.zeros((n_features, n_features))
    s_between = np.zeros((n_features, n_features))

    for label in range(len(labels_unique)):

        centered_vector_of_means = (means[label] - m0).reshape(-1, 1)
        s_between += probabilities[label] * np.matmul(centered_vector_of_means,
                                                      np.transpose(centered_vector_of_means))
        s_within += probabilities[label] * covariances[label]

    if drop_linear_dependent:  # VER DECIMASL!!!!!!
        _, index = _drop_linear_dependent_columns_and_rows(np.round(s_within, decimals=3), return_index=True)
        s_within = _drop_linear_dependent_columns_and_rows(s_within, index=index)
        s_between = _drop_linear_dependent_columns_and_rows(s_between, index=index)

    score = np.trace(np.matmul(pinv(s_within), s_between))

    return score


if __name__ == "__main__":
    '''
    Code to test function
    '''

    import pickle
    import bz2

    # Define constants
    LOAD_PATH = 'results/train/old/2018-11-13_17-45-56/data/'

    with bz2.BZ2File(LOAD_PATH + 'vehicles.bz2', 'rb') as h:
        contents = pickle.load(h)

    x = np.asarray(contents['results']['ohe_data'].todense())
    labels = contents['results']['labels']

    score_ = scatter_separability_score(x, labels, drop_linear_dependent=True)

    # gini = stats.gini_matrix(df, progress_bar=True, verbose=False)

    # threshold_range = sorted(set(gini.values.flatten().tolist()), reverse=True)



