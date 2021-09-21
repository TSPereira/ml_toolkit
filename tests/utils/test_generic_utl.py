import unittest
from ml_toolkit.utils.generic_utl import *


class MyTestCase(unittest.TestCase):
    def test_duplicated(self):
        self.assertEqual({1, 2}, duplicated([1, 2, 3, 1, 4, 5, 2]))

    def test_get_magnitude(self):
        self.assertEqual(get_magnitude(1000), 3)
        self.assertEqual(get_magnitude(-1e6), 6)
        self.assertTrue(np.array_equal(get_magnitude(np.array([10, 1000, -100, 0])), np.array([1, 3, 2, 0]),
                                       equal_nan=True))

    def test_ceil_decimal(self):
        self.assertEqual(np.array([1., 1., 1., -0., -0., -0.]).tolist(),
                         ceil_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6]).tolist())
        self.assertEqual(np.array([1., 1., 1., -1., -1., -1.]).tolist(),
                         ceil_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6], signed=False).tolist())

    def test_floor_decimal(self):
        self.assertEqual(np.array([0., 0., 0., -1., -1., -1.]).tolist(),
                         floor_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6]).tolist())
        self.assertEqual(np.array([0., 0., 0., -0., -0., -0.]).tolist(),
                         floor_decimal([0.5, 0.4, 0.6, -0.5, -0.4, -0.6], signed=False).tolist())


if __name__ == '__main__':
    unittest.main()
