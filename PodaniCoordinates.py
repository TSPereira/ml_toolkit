import numpy as np
from collections import Counter


class PodaniCoordinates:
    """
    Objective
    ---------
    This class implements an encoder that converts an ordinal feature to a set of coordinates such that the resulting
    Euclidean distance is equal to the metric presented by Podani (1999). The coordinates are obtained by linear algebra
    following ideas presented at the Stack Exchange website*.

    Usage
    -----
    By converting an ordinal feature to a set of coordinates it is possible to use these features in K-Means clustering.
    It is also relevant to mention that any transformation to these coordinates must imply only rotations or scaling.
    If this is not respected the relationship to the Podani metric is lost.

    References
    ----------
    Podani, J., 1999. Extending Gower's general coefficient of similarity to ordinal characters. Taxon, pp.331-340.

    * https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix

    Parameters
    ----------
    array: np.array, default: None
        numpy array with the ordinal feature vector to convert to Podani Coordinates

    object_order: list, default: None
        A list of unique elements that must include at least all the elements in 'array'. Its order is represents increasing
        ordinality.

    Attributes
    ----------
    STEP: todo
    ranks: todo
    elements_count: todo
    podani_matrix: todo
    n_objects: todo
    coordinates: todo
    shift: todo

    Example
    -------
    a = 3 * ['A'] + 4 * ['B'] + ['C'] + 2 * ['D'] + 3 * ['E']
    b = ['A', 'B', 'C', 'D', 'E']
    c = PodaniCoordinates()
    c.fit_transform(a, b)

    """

    def __init__(self, array=None, object_order=None):
        """Initialization of instances

        :param numpy.array|list array: ordinal feature vector to convert to Podani Coordinates
        :param list object_order: A list of unique elements that must include at least all the elements in 'array'. Its
        order is represents increasing ordinality.
        """

        self.STEP = 0.01

        self.array = None
        self.object_order = None

        self._validate_all(array, object_order)
        if array is not None and object_order is not None:
            self._match_inputs()

        self.ranks = dict()
        self.elements_count = dict()
        self.podani_matrix = None
        self.n_objects = None
        self.coordinates = None
        self.shift = None

    def _validate_all(self, array, object_order):
        if array is not None:
            self._validate_array(array)

        if object_order is not None:
            self._validate_object_order(object_order)

    def _validate_array(self, array):
        if isinstance(array, list):
            self.array = array
        else:
            try:
                self.array = array.tolist()
            except:
                ValueError('Array must be a list or a numpy array that can be converted to a list.')

    def _validate_object_order(self, object_order):
        if not isinstance(object_order, list):
            try:
                object_order = object_order.tolist()
            except:
                ValueError('Objected order must be a list or a numpy array that can be converted to a list.')
        if len(set(object_order)) < len(object_order):
            raise ValueError('There are duplicate items in the object order array.')
        self.object_order = object_order

    def _match_inputs(self):
        for element in set(self.array):
            if element not in self.object_order:
                ValueError('Element <' + str(element) + '> is not present in the list of ordered objects.')

    def _count_elements(self):
        self.elements_count = Counter(self.array)
        for element in self.object_order:
            if element not in self.elements_count.keys():
                self.elements_count[element] = 0

    def _get_ranks_of_objects(self):
        self._count_elements()
        self.ranks = dict()
        n = 0
        for element in self.object_order:
            rank = ((n + 1) + (n + self.elements_count[element])) / 2
            n = n + self.elements_count[element]
            self.ranks[element] = rank

    def _make_podani_matrix(self):
        # Build auxiliary variables to make the code more readable
        n = self.n_objects
        rank = dict()
        count = dict()

        for i in range(n):
            rank[i] = self.ranks[self.object_order[i]]
            count[i] = self.elements_count[self.object_order[i]]

        max_index, max_rank = (max(rank, key=rank.get), max(rank.values()))
        min_index, min_rank = (min(rank, key=rank.get), min(rank.values()))
        max_count = count[max_index]
        min_count = count[min_index]

        denominator = np.abs(max_rank - min_rank) - (max_count - 1) / 2 - (min_count - 1) / 2

        # Proper code to make the Podani matrix
        self.podani_matrix = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator = np.abs(rank[i] - rank[j]) - (count[i] - 1) / 2 - (count[j] - 1) / 2
                    self.podani_matrix[i, j] = numerator / denominator

    def _convert_distances_to_coordinates(self):
        # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix

        n = self.n_objects
        self.shift = -self.STEP

        eigenvalues = [-1]  # Value to force cycle to start
        eigenvectors = None

        while min(eigenvalues) < 0:  # Checks if Gram matrix is positive semi definite

            self.shift += self.STEP

            distance_squared_matrix = (self.podani_matrix + self.shift) / (1 + self.shift)

            for i in range(n):
                distance_squared_matrix[i, i] = 0

            gram_matrix = np.zeros([n, n])

            for i in range(n):
                for j in range(n):
                    gram_matrix[i, j] = (distance_squared_matrix[i, 0] + distance_squared_matrix[0, j] -
                                         distance_squared_matrix[i, j]) / 2

            eigenvalues, eigenvectors = np.linalg.eig(gram_matrix)

        root_eigenvalues = np.zeros(n)

        for i in range(n):
            root_eigenvalues[i] = (abs(eigenvalues[i]) ** (1 / 2))

        coords = np.matmul(eigenvectors, np.diag(root_eigenvalues))

        self.coordinates = dict()

        for i in range(self.n_objects):
            self.coordinates[self.object_order[i]] = coords[i]

    def _check_coordinates_correct(self, verbose=False):
        # Auxiliary function to check if coordinates match Podani matrix
        n = self.n_objects

        d = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                dist = self.coordinates[i, :] - self.coordinates[j, :]
                for k in range(n - 1):
                    d[i, j] += dist[k] ** 2

        d = d * (1 + self.shift) - self.shift
        for i in range(n):
            for j in range(n):
                if i == j:
                    d[i, j] = 0

        if verbose:
            print(d)
            print(self.podani_matrix)

        return np.allclose(d, self.podani_matrix)

    def fit(self, array=None, object_order=None):
        """Method to train the encoder. If no "array" or "object_order" is passed it will check if these were used to
        instantiate the class and use the existing parameters. If any of it is passed it will replace the existing ones

        :param numpy.array|list array: A list or numpy array of elements to train.
        :param list object_order: A list of unique elements that must include at least all the elements above.
                             Its order is represents increasing ordinality.
        """
        self._validate_all(array, object_order)
        if self.array is None or self.object_order is None:
            raise ValueError('Inputs are missing.')
        self._match_inputs()
        self._get_ranks_of_objects()
        self.n_objects = len(self.object_order)
        self._make_podani_matrix()
        self._convert_distances_to_coordinates()

    def transform(self, test_array):
        """Method to transform a test array using the trained encoder.

        :param numpy.array|list test_array: A list of elements to test.
        :return: An array with the coordinates of each element in the test set.
        """
        self._validate_array(test_array)
        for i, content in zip(range(len(test_array)), test_array):
            test_array[i] = self.coordinates[content]
        return test_array

    def fit_transform(self, array=None, object_order=None):
        """Method to train the encoder and transform the same array. If no "array" or "object_order" is passed it
        will check if these were used to instantiate the class and use the existing parameters. If any of it is
        passed it will replace the existing ones

        :param numpy.array|list array: A list of elements to train and test.
        :param list object_order: A list of unique elements that must include at least all the elements above.
                             Its order is represents increasing ordinality.
        :return: An array with the coordinates of each element in the test set.
        """
        self.fit(array, object_order)
        return self.transform(array)
