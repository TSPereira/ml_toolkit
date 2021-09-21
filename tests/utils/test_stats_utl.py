import unittest
from ml_toolkit.utils.stats_utl import *


class MyTestCase(unittest.TestCase):
    def test_moving_average(self):
        self.assertEqual(moving_average(list(range(10)), 3), [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(moving_average([3, 0, 3, 4, 3, 5, 0, 3, 4, 2, 0, 4, 3], 5),
                         [3.0, 1.5, 2.0, 2.5, 2.6, 3.0, 3.0, 3.0, 3.0, 2.8, 1.8, 2.6, 2.6])

        result = moving_average([3, 0, 3, 4, 3, 5, 0, 3, 4, 2, 0, 4, 3], 5, exclude_zeros=True)
        self.assertEqual(np.array(result).round(2).tolist(),
                         [3.0, 3.0, 3.0, 3.33, 3.25, 3.6, 3.6, 3.6, 3.8, 3.4, 3.4, 3.6, 3.2])

    def test_compute_mape(self):
        y = np.array([1, 0.9, 0.8, 0.7, 0.6])
        yhat = np.array([1, 0.9, 0.8, 0.7, 0.6])

        self.assertTrue(np.array_equal(compute_mape(y, yhat), np.array([0.])))
        self.assertTrue(np.array_equal(compute_mape(y, yhat, axis=1), np.array([0., 0., 0., 0., 0.])))

        yhat = np.array([1, 0.9, 0.4, 0.7, 0.6])
        self.assertTrue(np.array_equal(compute_mape(y, yhat), np.array([0.1])))
        self.assertTrue(np.array_equal(compute_mape(y, yhat, axis=1), np.array([0., 0., 0.5, 0., 0.])))

        y = np.array([1, 0.9, 0, 0.7, 0.6])
        yhat = np.array([1, 0.9, 0.8, 0.7, 0.6])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertWarns(UserWarning, compute_mape, y, yhat)
            self.assertTrue(np.array_equal(compute_mape(y, yhat).round(2), np.array([0.25])))

    def test_compute_wape(self):
        y = np.array([1, 0.9, 0.8, 0.7, 0.6])
        yhat = np.array([1, 0.9, 0.8, 0.7, 0.6])

        self.assertTrue(np.array_equal(compute_wape(y, yhat), np.array([0.])))
        self.assertTrue(np.array_equal(compute_wape(y, yhat, axis=1), np.array([0., 0., 0., 0., 0.])))

        yhat = np.array([1, 0.9, 0.4, 0.7, 0.6])
        self.assertTrue(np.array_equal(compute_wape(y, yhat).round(1), np.array([0.1])))
        self.assertTrue(np.array_equal(compute_wape(y, yhat, axis=1).round(1), np.array([0., 0., 0.5, 0., 0.])))

        y = np.array([1, 0.9, 0, 0.7, 0.6])
        yhat = np.array([1, 0.9, 0.8, 0.7, 0.6])
        self.assertTrue(np.array_equal(compute_wape(y, yhat).round(2), np.array([0.25])))

    def test_var_sparse(self):
        ...

    def test_std_sparse(self):
        ...

    def test_get_mean_confidence_interval(self):
        y = np.array([1, 2, 3, 4, 5])
        self.assertEqual(get_mean_confidence_interval(y), (3.0, 1.7604099353908769, 4.239590064609123))
        self.assertEqual(get_mean_confidence_interval(y, 0.9), (3.0, 1.9597032242488852, 4.040296775751115))

        y = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
        result = get_mean_confidence_interval(y)
        expected = (np.array([1.5, 3., 4.5, 6., 7.5]),
                    np.array([0.807, 1.614, 2.421, 3.228, 4.035]),
                    np.array([2.193, 4.386, 6.579, 8.772, 10.965]))

        for res, exp in zip(result, expected):
            self.assertTrue(np.array_equal(res.round(3), exp))


if __name__ == '__main__':
    unittest.main()
