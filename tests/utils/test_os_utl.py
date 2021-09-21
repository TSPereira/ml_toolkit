import unittest
import numpy as np
from ml_toolkit.utils.os_utl import *


class MyTestCase(unittest.TestCase):
    def test_check_types(self):
        @check_types(a=int, b=str, c=(int, str))
        def some_func(a, b, c):
            return type(a), type(b), type(c)

        self.assertEqual((int, str, int), some_func(1, '2', 1))
        self.assertEqual((int, str, str), some_func(1, '2', '1'))
        self.assertRaises(TypeError, some_func, '1', '2', 1)
        self.assertRaises(TypeError, some_func, 1, 2, 1)
        self.assertRaises(TypeError, some_func, 1, '2', [1])

    def test_check_options(self):
        @check_options(a=(1, 2), b=[4, 5])
        def some_func(a, b):
            return a, b

        self.assertRaises(KeyError, some_func, a=3, b=4)
        self.assertRaises(KeyError, some_func, a=2, b=3)
        self.assertEqual(some_func(1, 4), (1, 4))

    def test_check_interval(self):
        fail_suite = [(dict(items='a', min_value=0), dict(a=0)),
                      (dict(items='a', min_value=0), dict(a=-0.5)),
                      (dict(items='a', max_value=1), dict(a=1)),
                      (dict(items='a', max_value=1), dict(a=1.5)),
                      (dict(items=['a', 'b'], min_value=0), dict(b=-0.3)),
                      (dict(items=['a', 'b'], min_value=0, max_value=1), dict(b=-0.3))]

        success_suite = [(dict(items='a', min_value=0, closed='min'), dict(a=0)),
                         (dict(items='a', max_value=1, closed='max'), dict(a=1)),
                         (dict(items=['a', 'b'], min_value=0, max_value=1, closed='both'), dict(a=0, b=1))]

        def some_func(a=0.5, b=0.5):
            return

        for dec_params, func_params in fail_suite:
            f = check_interval(**dec_params)(some_func)
            self.assertRaises(ValueError, f, **func_params)

        for dec_params, func_params in success_suite:
            f = check_interval(**dec_params)(some_func)
            try:
                f(**func_params)
            except Exception as e:
                self.fail(f'some_func() raised {type(e)} unexpectedly!\n{e}')

    def test_filter_kwargs(self):
        def some_func(a=1, b=2):
            return

        suite = [(dict(a=10, c=1000), ['a']),
                 (dict(a=10, b=100, c=1000), ['a', 'b']),
                 (dict(c=1000), [])]

        for params, expected in suite:
            self.assertEqual(list(filter_kwargs(params, some_func).keys()), expected)

    def test_get_type_name(self):
        self.assertEqual('int', get_type_name(type(1)))
        self.assertEqual('str', get_type_name(type('1')))
        self.assertEqual('ndarray', get_type_name(type(np.array([0, 1, 2]))))

    def test_append_to_filename(self):
        self.assertEqual('test_option1.csv', append_to_filename('test.txt', 'option1', '.csv',
                                                                sep='_', remove_ext=True))
        self.assertEqual('test_option1.txt.csv', append_to_filename('test.txt', 'option1', '.csv',
                                                                    sep='_', remove_ext=False))
        self.assertEqual('test_option1.txt', append_to_filename('test.txt', 'option1', ext=None,
                                                                sep='_', remove_ext=False))
        self.assertEqual('test_option1', append_to_filename('test.txt', 'option1', ext=None,
                                                            sep='_', remove_ext=True))

    def test_is_valid_uuid(self):
        self.assertTrue(is_valid_uuid('3d1c4bda-faed-44aa-b34a-fa524ec0345a'))
        self.assertFalse(is_valid_uuid('3d1c4bda-faed-44aa-b34a-fa52'))

    def test_generate_uuids(self):
        uuids = generate_uuids(5)
        self.assertTrue(all(is_valid_uuid(uuid) for uuid in uuids))
        self.assertEqual(5, len(uuids))
        self.assertRaises(ValueError, generate_uuids, -10)


if __name__ == '__main__':
    unittest.main()
