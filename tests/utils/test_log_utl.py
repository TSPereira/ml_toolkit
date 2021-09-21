# -*- coding: utf-8 -*-
import os
import unittest
from io import StringIO
from time import sleep

from ml_toolkit.utils.log_utl import *


class CreateLoggerTestCase(unittest.TestCase):
    def test_output(self):
        name = 'test_output'
        logger = create_logger(name)

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, name)

        # cleanup for remaining tests
        logger.manager.loggerDict.pop(name)

    def test_handlers(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logger_ch = create_logger('test_ch')
            logger_fh = create_logger('test_fh', consolehandler=None, file='test_fh.log')
            logger_both = create_logger('test_both', file='test_both.log')
            logger_none = create_logger('test_none', consolehandler=None)

        # console handler
        self.assertEqual(len(logger_ch.handlers), 1)
        self.assertIsInstance(logger_ch.handlers[0], logging.StreamHandler)

        # file handler
        self.assertEqual(len(logger_fh.handlers), 1)
        self.assertIsInstance(logger_fh.handlers[0], logging.FileHandler)

        # both handlers
        self.assertEqual(len(logger_both.handlers), 2)
        self.assertIsInstance(logger_both.handlers[0], logging.StreamHandler)
        self.assertIsInstance(logger_both.handlers[1], logging.FileHandler)

        # None
        self.assertEqual(len(logger_none.handlers), 0)

        # close and remove log files created
        for logger in [logger_fh, logger_both]:
            logger.handlers[-1].close()
            os.remove(f'{logger.name}.log')

        # cleanup for remaining tests
        for name in ['test_ch', 'test_fh', 'test_both', 'test_none']:
            logging.Logger.manager.loggerDict.pop(name)

    def test_on_conflict(self):
        logger = create_logger('test_conflicts')

        # test raise
        self.assertRaises(KeyError, create_logger, 'test_conflicts')

        # test add
        self.assertWarns(UserWarning, create_logger, 'test_conflicts', on_conflict='add')
        self.assertEqual(len(logger.handlers), 2)

        # test override
        logger = create_logger('test_conflicts', on_conflict='override')
        self.assertEqual(len(logger.handlers), 1)

        # cleanup for remaining tests
        logger.manager.loggerDict.pop('test_conflicts')

    def test_format(self):
        fmt = '[TEST] %(message)s'
        datefmt = '%Y-%m-%d'

        # on the console handler
        logger = create_logger('test_format', fmt=fmt, datefmt=datefmt)
        self.assertEqual(logger.handlers[0].formatter.datefmt, datefmt)
        self.assertEqual(logger.handlers[0].formatter._fmt, fmt)

        # on the file handler
        logger = create_logger('test_format', consolehandler=None, file='test_format.log',
                               fmt=fmt, datefmt=datefmt, on_conflict='override')
        self.assertEqual(logger.handlers[0].formatter.datefmt, datefmt)
        self.assertEqual(logger.handlers[0].formatter._fmt, fmt)
        logger.handlers[0].close()
        os.remove('test_format.log')

        # on console handler with color
        logger = create_logger('test_format', fmt=fmt, datefmt=datefmt, colored=True, on_conflict='override')
        self.assertIsInstance(logger.handlers[0].formatter, ColorFormatter)

        # cleanup
        logger.manager.loggerDict.pop('test_format')

    def test_logger_formatter_monkeypatch(self):
        logger = create_logger('test_logger_formatter_monkeypatch', colored=True)
        self.assertTrue(logger.handlers[0].formatter.is_colored)

        logger.setFormattersIsColored(False)
        self.assertFalse(logger.handlers[0].formatter.is_colored)

        # cleanup
        logger.manager.loggerDict.pop('test_logger_formatter_monkeypatch')

    def test_levels(self):
        # check logger level and single handler level
        logger = create_logger('test_levels', logger_level=logging.WARNING, handlers_level=40)
        self.assertEqual(logger.level, logging.WARNING)
        self.assertEqual(logger.handlers[0].level, 40)

        # test both handlers level - single arg
        logger = create_logger('test_levels', file='test_levels.log', logger_level=logging.WARNING,
                               handlers_level=40, on_conflict='override')
        self.assertEqual(logger.handlers[0].level, 40)
        self.assertEqual(logger.handlers[1].level, 40)
        logger.handlers[1].close()

        # test both handlers level - tuple arg
        logger = create_logger('test_levels', file='test_levels.log', logger_level=logging.WARNING,
                               handlers_level=(40, 30), on_conflict='override')
        self.assertEqual(logger.handlers[0].level, 40)
        self.assertEqual(logger.handlers[1].level, 30)
        logger.handlers[1].close()

        # cleanup
        os.remove('test_levels.log')
        logger.manager.loggerDict.pop('test_levels')


class MonitorTestCase(unittest.TestCase):
    logger = create_logger('test_monitor', StringIO(), fmt='%(message)s', colored=False)

    def test_decorator(self):
        # setup
        mon = monitor(mode='time', logger=self.logger)

        @mon
        def some_func():
            sleep(0.1)

        # Run function
        some_func()

        expected = f'[func: {some_func.__qualname__}] took: {mon.duration}'
        result = self.logger.handlers[0].stream.getvalue().strip('\n').split('\n')[-1]
        self.assertEqual(result, expected)

    def test_context_manager(self):
        # Run function and get value from logger
        with monitor(mode='time', logger=self.logger) as m:
            ...

        expected = f'[block] took: {m.duration}'
        result = self.logger.handlers[0].stream.getvalue().strip('\n').split('\n')[-1]
        self.assertEqual(result, expected)

    def test_mode(self):
        # test only time
        with monitor(mode='time', logger=self.logger):
            ...
        result = self.logger.handlers[0].stream.getvalue().strip('\n').split('\n')[-1]
        self.assertNotIn('MB', result)
        self.assertIn('\u03BCs', result)

        # test only memory
        with monitor(mode='memory', logger=self.logger):
            ...
        result = self.logger.handlers[0].stream.getvalue().strip('\n').split('\n')[-1]
        self.assertNotIn('\u03BCs', result)
        self.assertIn('MB', result)

        # test both
        with monitor(mode='all', logger=self.logger):
            ...
        result = self.logger.handlers[0].stream.getvalue().strip('\n').split('\n')[-1]
        self.assertIn('\u03BCs', result)
        self.assertIn('MB', result)

    def test_disable_prefix(self):
        # Run function and get value from logger
        with monitor(mode='time', logger=self.logger, disable_prefix=True) as m:
            ...
        expected = f'took: {m.duration}'
        result = self.logger.handlers[0].stream.getvalue().strip('\n').split('\n')[-1]
        self.assertEqual(result, expected)

    def test_desc(self):
        # Run function and get value from logger
        with monitor(mode='time', logger=self.logger, desc='CustomDescription') as m:
            ...
        expected = f'[block: CustomDescription] took: {m.duration}'
        result = self.logger.handlers[0].stream.getvalue().strip('\n').split('\n')[-1]
        self.assertEqual(result, expected)

    def test_set_logger(self):
        # test with logger
        m = monitor(logger=self.logger)
        self.assertEqual(m.logger, self.logger)

        # test with StringIO
        m = monitor(logger=StringIO())
        self.assertIsInstance(m.logger, logging.Logger)
        self.assertEqual(m.logger.name, 'ml_toolkit.utils.log_utl.monitor')
        self.assertIsInstance(m.logger.handlers[0], logging.StreamHandler)

        # test with stdout
        m = monitor(logger=sys.stdout)
        self.assertIsInstance(m.logger, logging.Logger)
        self.assertEqual(m.logger.name, 'ml_toolkit.utils.log_utl.monitor')
        self.assertIsInstance(m.logger.handlers[0], logging.StreamHandler)

        # test with file
        m = monitor(logger='test.log')
        self.assertIsInstance(m.logger, logging.Logger)
        self.assertEqual(m.logger.name, 'ml_toolkit.utils.log_utl.monitor')
        self.assertIsInstance(m.logger.handlers[0], logging.FileHandler)
        m.logger.handlers[0].close()

        with open('test.log', 'w') as f:
            m = monitor(logger=f)
            self.assertIsInstance(m.logger, logging.Logger)
            self.assertEqual(m.logger.name, 'ml_toolkit.utils.log_utl.monitor')
            self.assertIsInstance(m.logger.handlers[0], logging.StreamHandler)
        os.remove('test.log')

        # test with not valid logger
        self.assertRaises(TypeError, monitor, logger=1)


class GeneralTestCase(unittest.TestCase):
    def test_wrap_text(self):
        self.assertEqual('abcdefghij\nklmnopqrst\nuvwyxz', wrap_text('abcdefghijklmnopqrstuvwyxz', 10))
        self.assertEqual('abcde\nfghij\nklmno\npqrst\nuvwyx\nz', wrap_text('abcdefghijklmnopqrstuvwyxz', 5))

    def test_time_diff(self):
        start_time = datetime(2021, 4, 17, 23, 11, 22, 829688)
        end_time = datetime(2021, 4, 18, 18, 15, 22, 100)
        self.assertEqual(time_diff(end_time, start_time, as_type='string'), '19h 3m 59s 170412\u03BCs')
        self.assertEqual(time_diff(end_time, start_time, as_type='tuple'), (19, 3, 59, 170412))

    def test_printv(self):
        def clear_stream(s):
            s.seek(0)
            s.truncate(0)

        suite = [('One', dict(verbose_level=2, verbose=1), ''),
                 ('Two', dict(verbose_level=2, verbose=2), 'Two'),
                 ('Three', dict(verbose_level=2, verbose=2, ident=1), '  Three'),
                 ('Four', dict(verbose_level=2, verbose=2, ident=2), '    Four'),
                 ('Five', dict(verbose_level=2, verbose=2, title=True), '\nFive\n----')]

        with redirect_stdout(StringIO()) as stream:
            for string, params, expected in suite:
                printv(string, **params)
                self.assertEqual(stream.getvalue().rstrip('\n'), expected)
                clear_stream(stream)

    def test_print_formatter(self):
        suite = [('this is a test', dict(ident=0, case_style='capitalize', title=False), 'This is a test'),
                 ('this is a test', dict(ident=1, case_style='title', title=False), '  This Is A Test'),
                 ('this is a test', dict(ident=2, case_style='upper', title=False), '    THIS IS A TEST'),
                 ('this is a test', dict(case_style='lower', title=False), 'this is a test'),
                 ('this is a test', dict(case_style='upper', title=True), '\nTHIS IS A TEST\n--------------\n'),
                 ('this is a test', dict(case_style=None, title=True), '\nThis Is A Test\n--------------\n')]

        for string, params, expected in suite:
            self.assertEqual(print_formatter(string, **params), expected)


if __name__ == '__main__':
    unittest.main()
