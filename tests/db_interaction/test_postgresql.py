import logging
import os
import unittest

import pandas as pd
from pandas.io.sql import DatabaseError
import psycopg2

from ml_toolkit.db_interaction.api import PostgreSQLManager
from ml_toolkit.utils.io_utl import get_decorators


print({k: v for k, v in os.environ.items() if k.startswith('POSTGRES')})
CFG = dict(user=os.environ.get('POSTGRES_USER', 'postgres'),
           password=os.environ.get('POSTGRES_PASSWORD', ''),
           host=os.environ.get('POSTGRES_HOST', 'localhost'),
           port=os.environ.get('POSTGRES_PORT', '5432'),
           database=os.environ.get('POSTGRES_DB', None))


def open_db():
    db = PostgreSQLManager(**CFG)
    db.logger.setLevel(logging.ERROR + 1)
    db.logger.setFormattersIsColored(False)
    db.set_exception_handling('raise')
    return db


class DBTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = open_db()
        cls.test_table = 'public.test_postgres'

    @classmethod
    def tearDownClass(cls):
        del cls.db


class ConnectionCase(unittest.TestCase):
    def test_connection(self):
        # to run locally you might need to edit the pg_hba.conf file to use "method" "trust" for local connections
        db = open_db()
        self.assertEqual(db.name, CFG.get('database') or CFG.get('user'))
        self.assertEqual(db.user, CFG.get('user'))

    def test_connection_fail(self):
        cfg = CFG.copy()
        cfg['user'] = 'lskdjfl'
        self.assertRaises(psycopg2.Error, PostgreSQLManager, **cfg)


class SchemaCase(DBTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_schemas = ('test1', 'test2', 'test3')

    def test_create_schema(self):
        for schema in self.test_schemas:
            self.db.create_schema(schema)

        self.db.create_schema(self.test_schemas[0], True)
        self.assertRaises(psycopg2.Error, self.db.create_schema, *(self.test_schemas[0], False))

    def test_get_schemas(self):
        self.assertIn('public', self.db.get_schemas())

    def test_drop_schemas(self):
        self.db.execute("CREATE TABLE test1.test ()")
        self.db.execute("CREATE TABLE test2.test ()")

        self.assertRaises(psycopg2.Error, self.db.drop_schema, *('test1', False))
        self.assertRaises(psycopg2.Error, self.db.drop_schema, *('smth', True, False))

        self.db.drop_schema('test1', True)
        self.assertRaises(psycopg2.Error, self.db.drop_schema, *(['test2', 'test3'], False))
        self.db.drop_schema(['test2', 'test3'], True)

    def test_set_active_schema(self):
        self.db.set_active_schema()
        self.assertEqual('public', self.db.active_schema)

        self.db.create_schema(self.test_schemas[0])
        self.db.set_active_schema(self.test_schemas[0])
        self.assertEqual(self.test_schemas[0], self.db.active_schema)
        self.db.drop_schema(self.test_schemas[0])

        self.db.set_active_schema('smth')
        self.assertEqual('public', self.db.active_schema)


class DropTableCase(DBTestCase):
    def setUp(self):
        self.db.execute(f'CREATE TABLE IF NOT EXISTS {self.test_table} ()')
        self.db.refresh()

    def tearDown(self):
        self.db.execute(f'DROP TABLE IF EXISTS {self.test_table} CASCADE')
        self.db.refresh()

    def test_drop(self):
        self.db.drop_table(self.test_table)
        self.assertNotIn(self.test_table, self.db.tables())

    def test_drop_multiple(self):
        self.db.execute('CREATE TABLE IF NOT EXISTS public.test_postgres1 ()')
        self.db.refresh()

        self.db.drop_table([self.test_table, 'public.test_postgres1'])
        self.assertNotIn(self.test_table, self.db.tables())
        self.assertNotIn('public.test_postgres1', self.db.tables())

    def test_if_not_exists(self):
        self.db.drop_table(self.test_table)
        self.assertRaises(psycopg2.Error, self.db.drop_table, self.test_table, if_exists=False)


class CreateEmptyTableCase(DBTestCase):
    def tearDown(self):
        self.db.drop_table(self.test_table)

    def test_new_table(self):
        self.db.create_empty_table(self.test_table, if_not_exists=True)
        self.assertIn(self.test_table, self.db.tables())
        self.assertTrue(self.db.query(f"SELECT * FROM {self.test_table}").empty)

    def test_if_not_exists(self):
        self.db.create_empty_table(self.test_table)
        self.assertRaises(psycopg2.Error, self.db.create_empty_table, self.test_table)
        self.db.create_empty_table(self.test_table, if_not_exists=True)

    def test_types_and_columns(self):
        params = dict()
        params['schema'], params['table_name'] = self.test_table.split('.')

        # types
        test_types = {'a': 'int', 'b': 'float', 'c': 'object'}
        types_query = f"""SELECT column_name,
                        CASE
                            WHEN domain_name IS NOT NULL THEN domain_name
                            WHEN data_type='character varying' THEN 'varchar('||character_maximum_length||')'
                            WHEN data_type='numeric' THEN 'numeric('||numeric_precision||','||numeric_scale||')'
                            ELSE data_type
                        END AS data_type
                        FROM information_schema.columns
                        WHERE table_schema = %(schema)s AND table_name = %(table_name)s """

        self.db.create_empty_table(self.test_table, types=test_types)
        self.assertEqual(self.db.query(types_query, params=params)['data_type'].tolist(),
                         ['integer', 'double precision', 'text'])

        # columns
        cols_query = f"""SELECT column_name FROM information_schema.columns 
                        WHERE table_schema = %(schema)s AND table_name = %(table_name)s """
        self.assertEqual(self.db.query(cols_query, params=params)['column_name'].to_list(), list(test_types))

    def test_types_from_df(self):
        params = dict()
        params['schema'], params['table_name'] = self.test_table.split('.')

        test_df = pd.DataFrame({'a': [1, 2, 3, 4, 5],
                                'b': [1.1, 2, 3.3, 4.4, 5.5],
                                'c': [1.1, 2, '3', 4, None]})
        test_types = {'b': 'object'}
        types_query = f"""SELECT column_name,
                        CASE
                            WHEN domain_name IS NOT NULL THEN domain_name
                            WHEN data_type='character varying' THEN 'varchar('||character_maximum_length||')'
                            WHEN data_type='numeric' THEN 'numeric('||numeric_precision||','||numeric_scale||')'
                            ELSE data_type
                        END AS data_type
                        FROM information_schema.columns
                        WHERE table_schema = %(schema)s AND table_name = %(table_name)s """

        self.db.create_empty_table(self.test_table, test_types, test_df)
        self.assertEqual(self.db.query(types_query, params=params)['data_type'].tolist(),
                         ['integer', 'text', 'text'])


class GetTableCase(DBTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.db.execute(f"CREATE TABLE {cls.test_table} (a integer, b text, c float)")
        cls.db.execute(f"INSERT INTO {cls.test_table} VALUES (1, 'b', 1), (1, 'a', 2.0), (2, 'c', null)")
        cls.db.refresh()

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_table(cls.test_table)
        super().tearDownClass()

    def test_select_all(self):
        df = self.db.get_table(self.test_table)
        self.assertEqual(df.shape[0], 3)
        self.assertEqual(df.shape[1], 3)

    def test_select_columns(self):
        cols_sets = (['a', 'b'], ['a', 'c'], ['b'], 'b')

        for cols in cols_sets:
            df = self.db.get_table(self.test_table, columns=cols)
            if isinstance(cols, str):
                cols = [cols]

            self.assertEqual(df.shape[0], 3)
            self.assertEqual(df.shape[1], len(cols))
            self.assertEqual(df.columns.to_list(), cols)

    def test_limit(self):
        limits_sets = (0, 1, 2, 3)

        for limit in limits_sets:
            df = self.db.get_table(self.test_table, limit=limit)
            self.assertEqual(df.shape[0], limit)

    def test_select_where(self):
        test_set = (dict(where="a = 1", result_set=[[1, 'b', 1], [1, 'a', 2.0]], shape=2),
                    dict(where="b = 'b'", result_set=[[1, 'b', 1]], shape=1),
                    dict(where="c is Null", result_set=[[2, 'c', None]], shape=1),
                    dict(where="a = 1 and b = 'b'", result_set=[[1, 'b', 1]], shape=1))

        for test in test_set:
            df = self.db.get_table(self.test_table, where=test['where'])
            self.assertEqual(df.shape[0], test['shape'])
            self.assertEqual(df.to_numpy(na_value=None).tolist(), test['result_set'])

    def test_where_safety(self):
        test_set = (f"a = 1; SELECT * FROM {self.test_table}",
                    f"'; SELECT * FROM {self.test_table} --")

        for test in test_set:
            self.assertRaises(DatabaseError, self.db.get_table, self.test_table, where=test)

    def test_order_and_sort(self):
        test_set = (('a', 'asc', [[1, 'b', 1], [1, 'a', 2.0], [2, 'c', None]]),
                    ('b', 'asc', [[1, 'a', 2.0], [1, 'b', 1], [2, 'c', None]]),
                    (['a', 'b'], 'asc', [[1, 'a', 2.0], [1, 'b', 1], [2, 'c', None]]),
                    # sort dir
                    ('a', 'desc', [[2, 'c', None], [1, 'b', 1], [1, 'a', 2.0]]),
                    ('b', 'desc', [[2, 'c', None], [1, 'b', 1], [1, 'a', 2.0]]),
                    (['a', 'b'], 'desc', [[2, 'c', None], [1, 'b', 1], [1, 'a', 2.0]]))

        for order, sort_dir, result in test_set:
            df = self.db.get_table(self.test_table, order_by=order, sort_dir=sort_dir)
            self.assertEqual(df.to_numpy(na_value=None).tolist(), result)


class UploadTableCase(DBTestCase):
    def tearDown(self):
        self.db.drop_table(self.test_table)

    def test__commit_table(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 5]})
        self.db.create_empty_table(self.test_table, from_df=values)
        self.db._commit_table(self.test_table, values.to_numpy().tolist(), values.columns.to_list())

        table = self.db.get_table(self.test_table)
        self.assertTrue(values.equals(table))

    def test_upload_columns(self):
        values = [[1, 2], [4, 5]]

        # raise error from creating a table without column definitions
        self.assertRaises(TypeError, self.db.upload_table, *(self.test_table, values, ['a', 'b']))

        # creates table without columns which results in error uploading data
        self.assertRaises(KeyError, self.db.upload_table, *(self.test_table, values, None))

    def test_upload_df(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 5]})
        self.db.upload_table(self.test_table, values)

        table = self.db.get_table(self.test_table)
        self.assertTrue(values.equals(table))

    def test_upload_values(self):
        values = [[1, 2], [4, 5]]
        columns = {'a': 'integer', 'b': 'float'}
        self.db.upload_table(self.test_table, values, columns)

        table = self.db.get_table(self.test_table)
        self.assertEqual(values, table.to_numpy().tolist())

    def test_upload_conflict(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 5]})
        self.db.upload_table(self.test_table, values)
        self.assertRaises(KeyError, self.db.upload_table, self.test_table, values, on_conflict='raise')
        self.db.upload_table(self.test_table, values, on_conflict='drop')


class ColumnsCase(DBTestCase):       
    def setUp(self):
        self.db.execute(f"CREATE TABLE {self.test_table} (a integer, b text, c float)")
        self.db.refresh()

    def tearDown(self) -> None:
        self.db.drop_table(self.test_table)

    def test_add_columns(self):
        self.db.add_columns(self.test_table, 'd')
        self.assertRaises(psycopg2.Error, self.db.add_columns, *(self.test_table, 'd'))
        self.db.add_columns(self.test_table, ['e', 'f'])
        self.db.add_columns(self.test_table, {'g': 'int', 'h': 'string'})

    def test_add_columns_not_null(self):
        schema, table_name = self.test_table.split('.')
        query = f"""SELECT column_name FROM information_schema.columns
                    WHERE table_schema = '{schema}'
                    AND table_name   = '{table_name}'
                    AND is_nullable = 'YES';"""

        self.db.add_columns(self.test_table, 'i', True)
        self.assertNotIn('i', self.db.query(query)['column_name'].to_list())

        self.db.add_columns(self.test_table, ['j', 'k'], True)
        self.assertNotIn('j', self.db.query(query)['column_name'].to_list())
        self.assertNotIn('k', self.db.query(query)['column_name'].to_list())

        self.db.add_columns(self.test_table, ['l', 'm'], [True, False])
        self.assertNotIn('l', self.db.query(query)['column_name'].to_list())
        self.assertIn('m', self.db.query(query)['column_name'].to_list())

        self.assertRaises(AssertionError, self.db.add_columns, *(self.test_table, ['n', 'o'], [True]))

    def test_alter_columns(self):
        self.db.execute(f"""INSERT INTO {self.test_table} VALUES (1, 'a', 1.0), (2, 'b', 4)""")

        self.db.alter_columns(self.test_table, {'a': 'float'})
        self.assertEqual(self.db.get_dtypes(self.test_table).to_dict(),
                         {'a': 'double precision', 'b': 'text', 'c': 'double precision'})

    def test_alter_columns_using(self):
        self.db.execute(f"""INSERT INTO {self.test_table} VALUES (1, '1', 1.0), ('2', '3', '4')""")

        # not using
        self.assertRaises(psycopg2.Error, self.db.alter_columns, *(self.test_table, {'b': 'integer'}))

        # using
        self.db.alter_columns(self.test_table, {'b': 'integer'}, using='integer')
        self.assertEqual(self.db.get_dtypes(self.test_table).to_dict(),
                         {'a': 'integer', 'b': 'integer', 'c': 'double precision'})

        # using multiple
        # setup all as text
        self.db.alter_columns(self.test_table, {'a': 'text', 'b': 'text', 'c': 'text'})

        # fail
        self.assertRaises(AssertionError, self.db.alter_columns,
                          *(self.test_table, {'a': 'integer', 'c': 'integer'}, ['integer']))
        self.assertRaises(psycopg2.Error, self.db.alter_columns,
                          *(self.test_table, {'a': 'integer', 'b': 'integer'}, ['integer', 'timestamp']))

        # convert multiple
        self.db.alter_columns(self.test_table, {'a': 'integer', 'b': 'integer'}, ['integer', 'integer'])
        self.assertEqual(self.db.get_dtypes(self.test_table).to_dict(),
                         {'a': 'integer', 'b': 'integer', 'c': 'text'})

        self.db.alter_columns(self.test_table, {'a': 'integer', 'b': 'integer', 'c': 'integer'}, 'integer')
        self.assertEqual(self.db.get_dtypes(self.test_table).to_dict(),
                         {'a': 'integer', 'b': 'integer', 'c': 'integer'})

    def test_drop_columns(self):
        self.assertRaises(psycopg2.Error, self.db.drop_columns, self.test_table, ['c', 'd'], if_exists=False)
        self.db.drop_columns(self.test_table, ['c', 'd'])
        self.assertNotIn('c', self.db.get_columns(self.test_table))

        self.db.drop_columns(self.test_table, 'a')
        self.assertNotIn('a', self.db.get_columns(self.test_table))

    def test_drop_columns_cascade(self):
        self.assertRaises(AssertionError, self.db.drop_columns, *(self.test_table, ['b', 'c'], [True]))

    def test_rename_column(self):
        self.db.rename_column(self.test_table, 'a', 'd')
        self.assertIn('d', self.db.get_columns(self.test_table))
        self.assertNotIn('a', self.db.get_columns(self.test_table))

        self.assertRaises(psycopg2.Error, self.db.rename_column, self.test_table, 'e', 'f')
        self.assertRaises(psycopg2.Error, self.db.rename_column, self.test_table, 'd', 'b')


class IndexCase(DBTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.db.create_schema('test1')
        cls.db.create_schema('test2')

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_schema(['test1', 'test2'])
        super().tearDownClass()

    def setUp(self):
        self.db.create_empty_table('test1.test', {'a': 'integer', 'b': 'float'})
        self.db.create_empty_table('test1.test1', {'a': 'integer', 'b': 'float'})
        self.db.create_empty_table('test2.test', {'a': 'integer', 'b': 'float'})

    def tearDown(self):
        self.db.drop_table(['test1.test', 'test1.test1', 'test2.test'])

    def test_create(self):
        self.db.create_index('test1.test', 'a')
        self.db.create_index('test1.test1', ['a', 'b'])
        self.assertRaises(psycopg2.Error, self.db.create_index, 'test2.test', 'c')

    def test_create_with_name(self):
        custom_index = 'custom_name'
        query = """SELECT * FROM pg_indexes WHERE schemaname != 'pg_catalog' 
                    AND schemaname = 'test1' AND tablename = 'test' """
        self.assertEqual(self.db.query(query).shape[0], 0)
        self.db.create_index('test1.test', 'a', custom_index)
        self.assertEqual(self.db.query(query).shape[0], 1)
        self.assertIn(custom_index, self.db.query(query)['indexname'].to_list())

    def test_create_unique(self):
        query = """SELECT * FROM pg_indexes WHERE schemaname != 'pg_catalog' 
                    AND schemaname = 'test1' AND tablename = 'test' """

        self.db.create_index('test1.test', 'a', unique=True)
        self.assertIn('unique', self.db.query(query).loc[0, 'indexdef'].lower())

    def test_create_non_unique(self):
        query = """SELECT * FROM pg_indexes WHERE schemaname != 'pg_catalog' 
                    AND schemaname = 'test1' AND tablename = 'test' """

        self.db.create_index('test1.test', 'a', unique=False)
        self.assertNotIn('unique', self.db.query(query).loc[0, 'indexdef'].lower())

    def test_create_on_conflict(self):
        self.db.create_index('test1.test', 'a')
        self.db.create_index('test1.test', ['a', 'b'])  # no conflict, works fine
        self.assertRaises(IndexError, self.db.create_index, 'test1.test', ['a', 'b'])
        self.db.create_index('test1.test', ['a', 'b'], on_conflict='drop')

    def test_drop(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.drop_index('test1.custom_name')

    def test_drop_cascade(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.drop_index('test1.custom_name', cascade=True)

    def test_drop_return_query(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.assertEqual(self.db.drop_index('test1.custom_name'), None)

        self.db.create_index('test1.test', 'a', 'custom_name')
        self.assertIsInstance(self.db.drop_index('test1.custom_name', return_query=True), str)

    def test_drop_no_schema(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.drop_index('custom_name')

    def test_drop_no_schema_multiple_same_name(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test2.test', 'a', 'custom_name')
        self.assertRaises(IndexError, self.db.drop_index, 'custom_name')
        self.db.drop_index('test2.custom_name')

    def test_get(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.assertIn('custom_name', self.db.get_index('custom_name')['indexname'].to_list())

    def test_get_on_schema(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test2.test', 'a', 'custom_name')

        self.assertEqual(self.db.get_index('custom_name').shape[0], 2)
        self.assertEqual(self.db.get_index('custom_name', 'test1').shape[0], 1)

    def test_get_all(self):
        idxs = (['test1', 'test', 'custom_name'],
                ['test1', 'test1', 'custom_name1'],
                ['test2', 'test', 'custom_name'])
        for schema, table, idx_name in idxs:
            self.db.create_index(f'{schema}.{table}', 'a', idx_name)

        idxs_read = self.db.get_indexes()[['schemaname', 'tablename', 'indexname']].to_numpy().tolist()
        for test in idxs:
            self.assertIn(test, idxs_read)

    def test_get_all_on_table_name(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test1.test1', 'a', 'custom_name1')
        self.db.create_index('test2.test', 'a', 'custom_name')
        self.assertEqual(self.db.get_indexes(table_name='test').shape[0], 2)
        self.assertEqual(self.db.get_indexes(table_name='test1').shape[0], 1)
        self.assertEqual(self.db.get_indexes(table_name='test3.test').shape[0], 0)
        self.assertEqual(self.db.get_indexes(table_name='test1.test').shape[0], 1)

    def test_get_all_on_schema(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test1.test1', 'a', 'custom_name1')
        self.db.create_index('test2.test', 'a', 'custom_name')
        self.assertEqual(self.db.get_indexes(schema='test1').shape[0], 2)
        self.assertEqual(self.db.get_indexes(schema='test2').shape[0], 1)

    def test_get_all_on_schema_and_table(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test1.test1', 'a', 'custom_name1')
        self.db.create_index('test2.test', 'a', 'custom_name')
        self.assertEqual(self.db.get_indexes(table_name='test', schema='test1').shape[0], 1)
        self.assertEqual(self.db.get_indexes(table_name='test1', schema='test1').shape[0], 1)
        self.assertEqual(self.db.get_indexes(table_name='test', schema='test2').shape[0], 1)
        self.assertEqual(self.db.get_indexes(table_name='test1.test', schema='test3').shape[0], 1)
        self.assertEqual(self.db.get_indexes(table_name='test3.test', schema='test1').shape[0], 0)

    def test_get_indexes_columns_by_name(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test1.test1', ['a', 'b'], 'custom_name1')
        self.assertEqual(self.db.get_indexes_columns('custom_name').values.tolist(), [['a']])
        self.assertEqual(self.db.get_indexes_columns('custom_name1').values.tolist(), [['a', 'b']])
        self.assertEqual(self.db.get_indexes_columns(['custom_name', 'custom_name1']).values.tolist(),
                         [['a'], ['a', 'b']])
        self.assertTrue(self.db.get_indexes_columns('some_other_name').empty)

    def test_get_indexes_columns_by_table(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test2.test', 'b', 'custom_name')
        self.db.create_index('test1.test1', ['a', 'b'], 'custom_name1')

        self.assertEqual(self.db.get_indexes_columns(table_name='test').values.tolist(), [['a'], ['b']])
        self.assertEqual(self.db.get_indexes_columns(table_name='test1').values.tolist(), [['a', 'b']])
        self.assertTrue(self.db.get_indexes_columns(table_name='test2').empty)

    def test_get_indexes_columns_by_schema(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test2.test', 'b', 'custom_name')
        self.db.create_index('test1.test1', ['a', 'b'], 'custom_name1')

        self.assertTrue(self.db.get_indexes_columns(schema='test').empty)
        self.assertEqual(self.db.get_indexes_columns(schema='test1').values.tolist(), [['a'], ['a', 'b']])
        self.assertEqual(self.db.get_indexes_columns(schema='test2').values.tolist(), [['b']])

    def test_get_indexes_columns_by_name_table(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test2.test', 'b', 'custom_name')
        self.db.create_index('test1.test1', ['a', 'b'], 'custom_name1')

        self.assertEqual(self.db.get_indexes_columns('custom_name', 'test').values.tolist(), [['a'], ['b']])
        self.assertEqual(self.db.get_indexes_columns('custom_name1', 'test1').values.tolist(), [['a', 'b']])
        self.assertEqual(self.db.get_indexes_columns(['custom_name', 'custom_name1'], 'test1').values.tolist(),
                         [['a', 'b']])
        self.assertTrue(self.db.get_indexes_columns('custom_name', 'test1').empty)
        self.assertTrue(self.db.get_indexes_columns(['custom_name', 'custom_name1'], 'test3').empty)

    def test_get_indexes_columns_by_name_schema(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test2.test', 'b', 'custom_name')
        self.db.create_index('test1.test1', ['a', 'b'], 'custom_name1')

        self.assertEqual(self.db.get_indexes_columns('custom_name', schema='test1').values.tolist(), [['a']])
        self.assertEqual(self.db.get_indexes_columns('custom_name1', schema='test1').values.tolist(), [['a', 'b']])
        self.assertEqual(self.db.get_indexes_columns(['custom_name', 'custom_name1'], schema='test2').values.tolist(),
                         [['b']])
        self.assertTrue(self.db.get_indexes_columns('custom_name', schema='test').empty)
        self.assertTrue(self.db.get_indexes_columns(['custom_name', 'custom_name1'], schema='test').empty)

    def test_get_indexes_columns_by_name_table_schema(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test1.test1', ['a', 'b'], 'custom_name1')

        self.assertEqual(self.db.get_indexes_columns('custom_name', table_name='test',
                                                     schema='test1').values.tolist(), [['a']])
        self.assertEqual(self.db.get_indexes_columns('custom_name1', table_name='test1',
                                                     schema='test1').values.tolist(), [['a', 'b']])
        self.assertTrue(self.db.get_indexes_columns('custom_name2', table_name='test', schema='test1').empty)

    def test_get_indexes_columns_by_table_schema(self):
        self.db.create_index('test1.test', 'a', 'custom_name')
        self.db.create_index('test2.test', 'b', 'custom_name')
        self.db.create_index('test1.test1', ['a', 'b'], 'custom_name1')

        self.assertEqual(self.db.get_indexes_columns(table_name='test', schema='test1').values.tolist(), [['a']])
        self.assertEqual(self.db.get_indexes_columns(table_name='test', schema='test2').values.tolist(), [['b']])
        self.assertEqual(self.db.get_indexes_columns(table_name='test1', schema='test1').values.tolist(), [['a', 'b']])
        self.assertTrue(self.db.get_indexes_columns(table_name='test1', schema='test').empty)


class PrimaryKeysCase(DBTestCase):
    def setUp(self) -> None:
        self.db.execute(f"CREATE TABLE {self.test_table} (a integer, b text PRIMARY KEY, c float)")
        self.db.execute(f"INSERT INTO {self.test_table} VALUES (1, 'b', 1), (3, 'a', 2.0), (2, 'c', null)")
        self.db.refresh()

    def tearDown(self) -> None:
        self.db.drop_table(self.test_table)

    def test_drop_primary_key(self):
        self.assertEqual(self.db.get_constraints(self.test_table, 'p').shape[0], 1)
        self.db.drop_primary_key(self.test_table)
        self.db.drop_primary_key(self.test_table)  # test it doesn't raise an error if it doesn't exist

    def test_get_primary_key(self):
        self.assertEqual(self.db.get_primary_key(self.test_table).shape[0], 1)
        self.db.drop_primary_key(self.test_table)
        self.assertEqual(self.db.get_primary_key(self.test_table).shape[0], 0)

    def test_get_primary_key_columns(self):
        self.assertEqual(self.db.get_primary_key_columns(self.test_table), ['b'])
        self.assertEqual(self.db.get_primary_key_columns(self.test_table, idx=True), [2])

        self.db.drop_primary_key(self.test_table)
        self.assertEqual(self.db.get_primary_key_columns(self.test_table), [])
        self.assertEqual(self.db.get_primary_key_columns(self.test_table, idx=True), [])

    def test_set_primary_key(self):
        self.db.drop_primary_key(self.test_table)
        self.assertEqual(self.db.get_primary_key(self.test_table).shape[0], 0)

        # set with one column
        self.db.set_primary_key(self.test_table, 'b')
        self.assertEqual(self.db.get_primary_key_columns(self.test_table), ['b'])

        # try to set another and catch error
        self.assertRaises(psycopg2.Error, self.db.set_primary_key, *(self.test_table, ['a', 'b']))

        # set with on_conflict='drop'
        self.db.set_primary_key(self.test_table, ['a', 'b'], on_conflict='drop')
        self.assertEqual(self.db.get_primary_key_columns(self.test_table), ['a', 'b'])

    def test_temporary_primary_key(self):
        keys = (['a'], ['a', 'b'])
        for key in keys:
            existing_key = self.db.get_primary_key_columns(self.test_table)

            with self.db._temporary_primary_key(key, self.test_table) as new_key:
                self.assertEqual(self.db.get_primary_key_columns(self.test_table), key)
                self.assertEqual(new_key, key)

            self.assertEqual(self.db.get_primary_key_columns(self.test_table), existing_key)

    def test_temporary_primary_key_no_existing_key(self):
        self.db.drop_primary_key(self.test_table)

        key = ['a']
        with self.db._temporary_primary_key(key, self.test_table) as new_key:
            self.assertEqual(self.db.get_primary_key_columns(self.test_table), key)
            self.assertEqual(new_key, key)

        self.assertEqual(self.db.get_primary_key_columns(self.test_table), [])

    def test_temporary_primary_key_conflict(self):
        key = ['a']
        existing_key = self.db.get_primary_key_columns(self.test_table)

        self.db.set_exception_handling('ignore')
        with self.db._temporary_primary_key(['a'], self.test_table) as new_key:
            self.db.execute(f"UPDATE {self.test_table} SET b = 'a' WHERE a = 1")
            self.assertEqual(self.db.get_primary_key_columns(self.test_table), key)
            self.assertEqual(new_key, key)

        self.assertNotEqual(self.db.get_primary_key_columns(self.test_table), existing_key)
        self.assertEqual(self.db.get_primary_key_columns(self.test_table), key)
        self.db.set_exception_handling('raise')


class MiscTableCase(DBTestCase):
    def setUp(self):
        self.db.execute(f"CREATE TABLE {self.test_table} (a integer, b text, c float)")
        self.db.execute(f"INSERT INTO {self.test_table} VALUES (1, 'b', 1), (1, 'a', 2.0), (2, 'c', null)")
        self.db.refresh()

    def tearDown(self):
        self.db.drop_table(self.test_table)

    def test_analyse_and_get_shape(self):
        self.assertEqual(self.db.get_shape(self.test_table, True), (3, 3))
        self.assertEqual(self.db.get_shape(self.test_table, False), (0, 3))

        # check that analyse is working and that the "exact" now gets the correct number of rows
        self.db.analyse(self.test_table)
        self.assertEqual(self.db.get_shape(self.test_table, False), (3, 3))

    def test_get_columns(self):
        self.assertEqual(self.db.get_columns(self.test_table), ['a', 'b', 'c'])

    def test_get_constraints(self):
        # set constraints
        self.db.execute(f"ALTER TABLE {self.test_table} ADD PRIMARY KEY (b)")
        self.db.execute(f"ALTER TABLE {self.test_table} ADD UNIQUE (c)")

        self.assertEqual(self.db.get_constraints(self.test_table).shape[0], 2)
        self.assertEqual(self.db.get_constraints(self.test_table)['contype'].to_list(), ['p', 'u'])

        self.assertEqual(self.db.get_constraints(self.test_table, 'primary').shape[0], 1)
        self.assertNotIn('u', self.db.get_constraints(self.test_table, 'p')['contype'].to_list())

    def test_get_dtypes(self):
        test_set = ((None, {'a': 'integer', 'b': 'text', 'c': 'double precision'}),
                    (['a', 'b'], {'a': 'integer', 'b': 'text'}),
                    (['a', 'b', 'd'], {'a': 'integer', 'b': 'text'}),
                    (['a'], {'a': 'integer'}),
                    ('a', {'a': 'integer'}))

        for columns, expected in test_set:
            self.assertEqual(self.db.get_dtypes(self.test_table, columns=columns).to_dict(), expected)

    def test_get_na(self):
        self.db.analyse(self.test_table)
        self.assertEqual(self.db.get_na(self.test_table).to_dict(), {'a': 0, 'b': 0, 'c': 1})
        self.assertEqual(self.db.get_na(self.test_table, ['a', 'b']).to_dict(), {'a': 0, 'b': 0})
        self.assertEqual(self.db.get_na(self.test_table, 'a').to_dict(), {'a': 0})
        self.assertEqual(self.db.get_na(self.test_table, ['a', 'b', 'd']).to_dict(), {'a': 0, 'b': 0})

        na = self.db.get_na(self.test_table, relative=True).round(5)
        expected = pd.Series({'a': 0.0, 'b': 0.0, 'c': 1/3}).round(5)
        self.assertTrue(na.equals(expected))

    def test_get_nunique(self):
        self.assertEqual(self.db.get_nunique(self.test_table).to_dict(), {'a': 2, 'b': 3, 'c': 2})
        self.assertEqual(self.db.get_nunique(self.test_table, count_null=True).to_dict(), {'a': 2, 'b': 3, 'c': 3})
        self.assertEqual(self.db.get_nunique(self.test_table, ['a', 'b']).to_dict(), {'a': 2, 'b': 3})
        self.assertEqual(self.db.get_nunique(self.test_table, 'a').to_dict(), {'a': 2})
        self.assertEqual(self.db.get_nunique(self.test_table, ['a', 'b', 'd']).to_dict(), {'a': 2, 'b': 3})

    def test_get_summary(self):
        self.db.analyse(self.test_table)
        summary = self.db.get_summary(self.test_table, count_null=True).round(5)
        expected = pd.DataFrame([['integer', 2, 0, 0.0],
                                 ['text', 3, 0, 0.0],
                                 ['double precision', 3, 1, 1/3]],
                                columns=['type', 'distinct', 'missing_values', 'missing_values_per'],
                                index=['a', 'b', 'c']).round(5)
        self.assertTrue(summary.equals(expected))

        summary = self.db.get_summary(self.test_table, count_null=False).round(5)
        expected = pd.DataFrame([['integer', 2, 0, 0.0],
                                 ['text', 3, 0, 0.0],
                                 ['double precision', 2, 1, 1/3]],
                                columns=['type', 'distinct', 'missing_values', 'missing_values_per'],
                                index=['a', 'b', 'c']).round(5)
        self.assertTrue(summary.equals(expected))

    def test_rename_table(self):
        new_name = 'public.test_postgres_new'
        self.db.drop_table(new_name)

        self.db.rename_table(self.test_table, new_name)
        self.assertIn(new_name, self.db.tables())

        # check if exists
        self.assertRaises(psycopg2.Error, self.db.rename_table, 'smth', 'smth_new')
        self.db.rename_table('smth', 'smth_new', True)

        self.db.rename_table(new_name, self.test_table)
        self.assertNotIn(new_name, self.db.tables())


class DeleteRowsCase(DBTestCase):
    def setUp(self):
        self.db.execute(f"CREATE TABLE {self.test_table} (a integer, b text, c float)")
        self.db.execute(f"INSERT INTO {self.test_table} VALUES (1, 'b', 1), (1, 'a', 2.0), (2, 'c', null)")
        self.db.refresh()

    def tearDown(self):
        self.db.drop_table(self.test_table)

    def test_delete_all(self):
        self.db.delete_rows(self.test_table)
        self.assertEqual(self.db.get_shape(self.test_table)[0], 0)

    def test_delete_where_single(self):
        test_sets = (dict(where="b = 'a'", col='b', result='a', shape=2),
                     dict(where="b = 'a'", col='b', result='a', shape=2),  # repetition. shouldn't do anything
                     dict(where="c is Null", col='c', result=None, shape=1),
                     dict(where="a = 1", col='a', result='a', shape=0))

        for test in test_sets:
            self.db.delete_rows(self.test_table, where=test['where'])
            self.assertNotIn(test['result'], self.db.get_table(self.test_table)[test['col']].to_list())
            self.assertEqual(self.db.get_table(self.test_table).shape[0], test['shape'])

    def test_delete_where_multiple(self):
        self.db.delete_rows(self.test_table, where="a = 1")
        self.assertNotIn(1, self.db.get_table(self.test_table)['a'].to_list())
        self.assertEqual(self.db.get_table(self.test_table).shape[0], 1)

    def test_delete_where_multiple_complex(self):
        self.db.delete_rows(self.test_table, where="a = 1 and b = 'a' ")
        self.assertNotIn([1, 'a'], self.db.get_table(self.test_table)[['a', 'b']].to_numpy().tolist())
        self.assertEqual(self.db.get_table(self.test_table).shape[0], 2)

    def test_check_where_safety(self):
        test_set = (f"a = 1; SELECT * FROM {self.test_table}",
                    f"'; SELECT * FROM {self.test_table} --")

        for test in test_set:
            self.assertRaises(DatabaseError, self.db.delete_rows, self.test_table, where=test)


class CopyTableCase(DBTestCase):
    def setUp(self):
        self.db.upload_table(self.test_table, pd.DataFrame({'a': [1, 3, 2], 'b': [4, 5, 6], 'c': [0, 0, 0]}))
        self.new_table_name = 'public.test_postgres1'

    def tearDown(self):
        self.db.drop_table(self.test_table)
        self.db.drop_table(self.new_table_name)

    def test_copy_all(self):
        self.db.copy_table(self.test_table, self.new_table_name)
        self.assertTrue(self.db.get_table(self.test_table).equals(self.db.get_table(self.new_table_name)))

    def test_columns(self):
        self.db.copy_table(self.test_table, self.new_table_name, columns=['a', 'b'])
        old = self.db.get_table(self.test_table, columns=['a', 'b'])
        new = self.db.get_table(self.new_table_name)
        self.assertTrue(new.equals(old))

    def test_where(self):
        self.db.copy_table(self.test_table, self.new_table_name, where="a in (1, 2)")
        old = self.db.get_table(self.test_table, where="a in (1, 2)")
        new = self.db.get_table(self.new_table_name)
        self.assertTrue(new.equals(old))

    def test_structure_only(self):
        self.db.copy_table(self.test_table, self.new_table_name, structure_only=True)

        old_columns = self.db.get_columns(self.test_table)
        new_columns = self.db.get_columns(self.new_table_name)
        self.assertEqual(old_columns, new_columns)

        old_dtypes = self.db.get_dtypes(self.test_table)
        new_dtypes = self.db.get_dtypes(self.new_table_name)
        self.assertTrue(old_dtypes.equals(new_dtypes))

        self.assertTrue(self.db.get_table(self.new_table_name).empty)

    def test_another_schema(self):
        self.db.create_schema('test1')
        self.db.copy_table(self.test_table, 'test1', destination_schema='test1')
        self.assertIn('test1.test1', self.db.tables('test1'))
        self.assertTrue(self.db.get_table(self.test_table).equals(self.db.get_table('test1.test1')))
        self.db.drop_schema('test1', cascade=True)


class AppendTableCase(DBTestCase):
    def setUp(self):
        self.db.create_empty_table(self.test_table, {'a': 'integer', 'b': 'float'})

    def tearDown(self):
        self.db.drop_table(self.test_table)

    def test__check_integrity(self):
        # DataFrame checks
        self.db._check_integrity(pd.DataFrame({'a': [1, 2], 'b': [4, 5]}), ['a', 'b'])
        self.db._check_integrity(pd.DataFrame({'a': [1, 2], 'b': [4, 5]}), ['a'])
        self.db._check_integrity(pd.DataFrame({'a': [1, 2], 'b': [4, 5]}), None)

        # list of lists checks
        self.db._check_integrity([[1, 2], [4, 5]], ['a', 'b'])
        self.assertRaises(ValueError, self.db._check_integrity, [[1, 2], [4, 5]], ['a'])
        self.assertRaises(ValueError, self.db._check_integrity, [[1, 2], [4, 5]], 'a')

    def test__update_table_schema_df_columns(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 5]})
        column_sets = (['a'], ['a', 'b'], ['a', 'b', 'c'], {'a': 'integer', 'd': 'text'})

        for columns in column_sets:
            _values, _columns = self.db._update_table_schema(self.test_table, values, columns)
            self.assertEqual(values.to_numpy().tolist(), _values)
            self.assertEqual(values.columns.to_list(), _columns)

    def test__update_table_schema_df_on_new_columns(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 5], 'c': [6, 7]})

        # 'raise' doesn't filter the dataframe and the error will be raised when the stmt is executed
        _values, _columns = self.db._update_table_schema(self.test_table, values, [], 'raise')
        self.assertEqual(values.to_numpy().tolist(), _values)
        self.assertEqual(values.columns.to_list(), _columns)

        _values, _columns = self.db._update_table_schema(self.test_table, values, [], 'ignore')
        self.assertEqual(values[['a', 'b']].to_numpy().tolist(), _values)
        self.assertEqual(['a', 'b'], _columns)

        _values, _columns = self.db._update_table_schema(self.test_table, values, [], 'add')
        self.assertEqual(values.to_numpy().tolist(), _values)
        self.assertEqual(values.columns.to_list(), _columns)
        self.assertIn('c', self.db.get_columns(self.test_table))

    def test__update_table_schema_sequence_columns(self):
        values = [[1, 2], [4, 5]]
        column_sets = (['a', 'b'], {'a': 'integer', 'd': 'text'})

        for columns in column_sets:
            _values, _columns = self.db._update_table_schema(self.test_table, values, columns)
            self.assertEqual(values, _values)
            self.assertEqual(list(columns), _columns)

    def test__update_table_schema_sequence_on_new_columns(self):
        values = [[1, 2], [4, 5]]

        # no new column definition
        self.assertRaises(ValueError, self.db._update_table_schema, self.test_table, values, ['a', 'c'], 'add')

        # new column - raise
        columns = {'a': 'integer', 'd': 'text'}
        _values, _columns = self.db._update_table_schema(self.test_table, values, columns, 'raise')
        self.assertEqual(values, _values)
        self.assertEqual(list(columns), _columns)

        # new column - ignore
        columns = {'a': 'integer', 'd': 'text'}
        _values, _columns = self.db._update_table_schema(self.test_table, values, columns, 'ignore')
        self.assertEqual(values, _values)
        self.assertEqual(list(columns), _columns)

        # new column - add
        columns = {'a': 'integer', 'd': 'text'}
        _values, _columns = self.db._update_table_schema(self.test_table, values, columns, 'add')
        self.assertEqual(values, _values)
        self.assertEqual(list(columns), _columns)
        self.assertIn('d', self.db.get_columns(self.test_table))

    def test_append_new_table(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 5]})
        self.db.append_to_table('public.test_postgres1', values)
        table = self.db.get_table('public.test_postgres1')
        self.assertTrue(values.equals(table))
        self.db.drop_table('public.test_postgres1')

    def test_append_new_table_no_column_definition(self):
        values = [[1, 2], [4, 5]]
        self.assertRaises(TypeError, self.db.append_to_table, 'public.test_postgres1', values, ['a', 'b'])
        self.assertRaises(TypeError, self.db.append_to_table, 'public.test_postgres2', values, None)

    def test_append_to_table(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 5]})
        self.db.append_to_table(self.test_table, values)


class UpdateTableCase(DBTestCase):
    def setUp(self):
        self.db.upload_table(self.test_table, pd.DataFrame({'a': [1, 3, 2], 'b': [4, 5, 6], 'c': [0, 0, 0]}))

    def tearDown(self):
        self.db.drop_table(self.test_table)

    def test_table_existence(self):
        self.assertRaises(psycopg2.Error, self.db.update_table, 'public.test_smth', ['b'], 1)
        self.db.update_table(self.test_table, ['b'], 1)

    def test_integrity(self):
        self.assertRaises(IndexError, self.db.update_table, self.test_table, ['b', 'a'], 1)
        self.assertRaises(IndexError, self.db.update_table, self.test_table, ['b', 'a'], [2])
        self.assertRaises(IndexError, self.db.update_table, self.test_table, 'b', [1, 2])
        self.assertRaises(ValueError, self.db.update_table, self.test_table, [], [])

    def test_update(self):
        self.db.update_table(self.test_table, 'b', 1)
        self.assertEqual(self.db.get_table(self.test_table, 'b')['b'].unique().tolist(), [1])

        self.db.update_table(self.test_table, ['b'], 3)
        self.assertEqual(self.db.get_table(self.test_table, 'b')['b'].unique().tolist(), [3])

    def test_with_expressions(self):
        self.db.update_table(self.test_table, ['b'], ['b+3'])
        self.assertEqual(self.db.get_table(self.test_table, 'b')['b'].to_list(), [7, 8, 9])

        self.db.update_table(self.test_table, ['b', 'c'], [2, 'a+b'])
        self.assertEqual(self.db.get_table(self.test_table, ['b', 'c']).to_numpy().tolist(), [[2, 8], [2, 11], [2, 11]])

    def test_where(self):
        self.db.update_table(self.test_table, 'b', 1, where='a=1')
        self.assertEqual(self.db.get_table(self.test_table).to_numpy().tolist(), [[3, 5, 0], [2, 6, 0], [1, 1, 0]])

        self.db.update_table(self.test_table, ['b', 'c'], [3, 5], where='a != 1')
        self.assertEqual(self.db.get_table(self.test_table).to_numpy().tolist(), [[1, 1, 0], [3, 3, 5], [2, 3, 5]])

    def test_safety(self):
        injection = "SELECT * FROM public.test --"
        self.assertRaises(DatabaseError, self.db.update_table, self.test_table, 'a', 1, where=f"b=4; {injection}")
        self.assertRaises(DatabaseError, self.db.update_table, self.test_table, 'a', 1, where=f"'; {injection}")
        self.assertRaises(DatabaseError, self.db.update_table, self.test_table, 'a', f"1; {injection}")
        self.assertRaises(DatabaseError, self.db.update_table, self.test_table, ['a', 'b'], [f"1; {injection}", 2])


class UpsertTableCase(DBTestCase):
    def setUp(self):
        self.db.upload_table(self.test_table, pd.DataFrame({'a': [1, 3, 2], 'b': [4, 5, 6], 'c': [0, 0, 0]}))

    def tearDown(self):
        self.db.drop_table(self.test_table)

    def test_upsert_new_table(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 6], 'c': [1, 3]})
        self.db.upsert_table('public.test_postgres1', values)
        self.db.drop_table('public.test_postgres1')

    def test_upsert_no_pkey(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [4, 6], 'c': [1, 3]})
        self.assertRaises(KeyError, self.db.upsert_table, self.test_table, values)

    def test_upsert_no_pkey_existing_pkey(self):
        values = pd.DataFrame({'a': [7, 1, 4], 'b': [5, 6, 7], 'c': [1, 3, 4]})

        self.db.set_primary_key(self.test_table, 'b')
        self.db.upsert_table(self.test_table, values)
        expected = pd.DataFrame({'a': [1, 7, 1, 4], 'b': [4, 5, 6, 7], 'c': [0, 1, 3, 4]})
        self.assertTrue(expected.equals(self.db.get_table(self.test_table)))

    def test_upsert_new_pkey(self):
        values = pd.DataFrame({'a': [7, 1, 3], 'b': [5, 6, 7], 'c': [1, 3, 4]})

        self.db.upsert_table(self.test_table, values, id_column_pkey='a')
        expected = pd.DataFrame({'a': [2, 7, 1, 3], 'b': [6, 5, 6, 7], 'c': [0, 1, 3, 4]})
        self.assertTrue(expected.equals(self.db.get_table(self.test_table)))

        values = pd.DataFrame({'a': [5, 1, 4], 'b': [5, 6, 7], 'c': [1, 3, 4]})
        self.db.upsert_table(self.test_table, values, id_column_pkey=['c'])
        expected = pd.DataFrame({'a': [2, 5, 1, 4], 'b': [6, 5, 6, 7], 'c': [0, 1, 3, 4]})
        self.assertTrue(expected.equals(self.db.get_table(self.test_table)))


class MiscCase(DBTestCase):
    def test_methods_parse_schema_table(self):
        methods = ['analyse', 'add_columns', 'alter_columns', 'drop_columns', 'rename_column', 'create_index',
                   'drop_primary_key', 'get_primary_key', 'get_primary_key_columns', 'set_primary_key',
                   'append_to_table', 'delete_rows', 'copy_table', 'create_empty_table', 'get_table',
                   'rename_table', 'update_table', 'upload_table', 'upsert_table', 'get_columns', 'get_constraints',
                   'get_dtypes', 'get_na', 'get_nunique', 'get_shape', 'get_summary', '_commit_table',
                   '_update_table_schema']

        decorators = get_decorators(PostgreSQLManager)
        methods_registered = [k for k, v in decorators.items() if 'parse_schema_table' in v]
        self.assertEqual(sorted(methods), sorted(methods_registered))

    def test_get_transactions(self):
        self.assertTrue(not self.db.get_transactions().empty)

    def test_get_transactions_state(self):
        states = set(self.db.get_transactions('active')['state'].to_list())
        self.assertEqual({'active'}, states)


if __name__ == '__main__':
    unittest.main()
