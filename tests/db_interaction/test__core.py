import logging
import unittest
import warnings
from io import StringIO

import pandas as pd
from ml_toolkit.utils.log_utl import create_logger
from ml_toolkit.db_interaction._core import join_schema_to_table_name, parse_schema_table

from .test_postgresql import DBTestCase


# SOME METHODS NEED TO BE TESTED WITH AN ACTIVE CONNECTION
# USING POSTGRESQL
class PgsqlDBTestCase(DBTestCase):
    ...


class BaseActorSQLCase(PgsqlDBTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_table = 'public.test_base_actor_sql'
        with cls.db.conn.cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {cls.test_table} (a integer, b integer, c integer)")
            cls.db.conn.commit()
            cls.db.refresh()

    @classmethod
    def tearDownClass(cls):
        with cls.db.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {cls.test_table} CASCADE")
            cls.db.conn.commit()
            cls.db.refresh()

        super().tearDownClass()

    def test_cursor_manager_exception(self):
        with self.assertRaises(self.db._cursor_manager_exceptions[-1]), self.db.cursor_manager() as cur:
            cur.execute('some wrong sql stmt')

    def test_set_active_schema(self):
        self.db.set_active_schema('test')
        self.assertEqual(self.db.active_schema, 'public')

        new_schema = 'test_baseactor'
        self.db.execute(f"CREATE SCHEMA {new_schema}")
        self.db.set_active_schema(new_schema)
        self.assertEqual(self.db.active_schema, new_schema)

        self.db.set_active_schema()
        self.assertEqual(self.db.active_schema, 'public')

        self.db.execute(f"DROP SCHEMA {new_schema}")

    def test_tables(self):
        self.assertIsInstance(self.db.tables(), list)
        self.assertIn(self.test_table, self.db.tables())

        self.assertIsInstance(self.db.tables('public'), list)
        self.assertIn(self.test_table, self.db.tables('public'))

        self.assertEqual(self.db.tables('smth'), [])

    def test_query(self):
        output = self.db.query("SELECT current_database()")
        self.assertEqual(output['current_database'].to_list(), [self.db.name])
        self.assertIsInstance(output, pd.DataFrame)

    def test_execute_stmt(self):
        new_table = "public.execute_stmt"
        self.db.execute(f"CREATE TABLE IF NOT EXISTS {new_table} ()")
        self.db.refresh()
        self.assertIn(new_table, self.db.tables())

        self.db.execute(f"DROP TABLE IF EXISTS {new_table} CASCADE")
        self.db.refresh()
        self.assertNotIn(new_table, self.db.tables())

    def test_execute_params(self):
        query_get_table = f'SELECT * FROM {self.test_table}'
        params = dict(values=(10, 20, 30))
        self.db.execute(f"INSERT INTO {self.test_table} VALUES %(values)s", params)
        self.assertIn(list(params['values']), self.db.query(query_get_table).to_numpy().tolist())

        self.db.execute(f"DELETE FROM {self.test_table} WHERE (a, b, c) = %s", (params['values'], ))
        self.assertNotIn(list(params['values']), self.db.query(query_get_table).to_numpy().tolist())

    def test_execute_return_cursor_metadata(self):
        values = ((1, 2, 3), (4, 5, 6))
        cur_meta = self.db.execute(f"INSERT INTO {self.test_table} VALUES %s, %s", values, return_cursor_metadata=True)
        self.assertEqual(cur_meta.rowcount, 2)

        query = f"DELETE FROM {self.test_table}"
        cur_meta = self.db.execute(query, return_cursor_metadata=True)
        self.assertEqual(cur_meta.query.decode(), query)

    def test_execute_log(self):
        msg = 'Executed successfully.'
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            create_logger(self.db.logger.name, StringIO(), fmt='%(message)s', colored=False, on_conflict='add')
        self.db.execute("SELECT current_database()", log=msg)

        handler = self.db.logger.handlers[-1].stream
        handler.seek(0)
        self.assertEqual(handler.read().strip(), msg)
        self.db.logger.handlers.pop(-1)

    def test_execute_debug(self):
        self.db.toggle_debug()
        query = "CREATE SCHEMA IF NOT EXISTS test_schema"
        self.assertEqual(self.db.execute(query), query)
        self.db.toggle_debug()

    def test_get_sql(self):
        db = super(self.db.__class__, self.db)
        stmt = 'INSERT INTO public.test VALUES ?'
        values = (1, 2, 3)
        self.assertEqual(db.get_sql(stmt, (values, )), stmt[:-1]+str(values))

        stmt = 'INSERT INTO public.test VALUES $, $'
        values = ((1, 2, 3), (4, 5, 6))
        self.assertEqual(db.get_sql(stmt, values, params_symbol='$'), stmt[:-4]+', '.join(str(val) for val in values))


class MiscCase(unittest.TestCase):
    def test_join_schema_to_table_name(self):
        self.assertEqual(join_schema_to_table_name('test', 'public'), 'public.test')
        self.assertEqual(join_schema_to_table_name('public.test', 'some_other_schema'), 'public.test')
        self.assertEqual(join_schema_to_table_name('test', None), 'test')
        self.assertEqual(join_schema_to_table_name('public.test', None), 'public.test')

        self.assertRaises(TypeError, join_schema_to_table_name, 'test')

    def test_parse_schema_table_correct_signature(self):
        @parse_schema_table
        def test(table_name, schema=None):
            return table_name, schema

        expected = ('test', 'public')
        self.assertEqual(test('test', 'public'), expected)
        self.assertEqual(test('public.test'), expected)
        self.assertEqual(test('public.test', 'some_other_schema'), expected)
        self.assertEqual(test('test'), expected)

    def test_parse_schema_table_acceptable_signature(self):
        @parse_schema_table
        def test(table_name, **kwargs):
            return table_name, kwargs['schema']

        expected = ('test', 'public')
        self.assertEqual(test('test', schema='public'), expected)
        self.assertEqual(test('public.test'), expected)
        self.assertEqual(test('public.test', schema='some_other_schema'), expected)
        self.assertEqual(test('test'), expected)

    def test_parse_schema_table_unacceptable_signature(self):
        @parse_schema_table
        def test_fail(table_name):
            return table_name

        self.assertRaises(TypeError, test_fail, 'test')
        self.assertRaises(TypeError, test_fail, 'public.test')
