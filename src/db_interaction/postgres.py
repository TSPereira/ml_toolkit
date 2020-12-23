import warnings
from functools import partial
from itertools import starmap
from contextlib import contextmanager

import pandas as pd
import pandas.io.sql
import psycopg2
from psycopg2.extras import execute_values
import io

from .base import BaseActor, prepare_table_name, join_schema_to_table_name
from ..utils.os_utl import check_types, check_options, NoneType


class PostgreSQLManager(BaseActor):
    """This class provides methods to interact with an Oracle Database

    Arguments:
        :arg user: User identification in the oracle database
        :arg password: User password to the oracle database
        :arg name: Name of the database to connect to
        :arg host: Address to the oracle database (if 'name' is not provided)
        :arg port: Port to connect through (if 'name' is not provided)
        :arg kwargs: Additional keyword arguments to pass to BaseActor class (active_schema)

    """
    def __init__(self, user, password, name=None, host=None, port=None, **kwargs):
        self.host = host
        self.port = port
        self._flavor = 'PostGreSQL'

        super().__init__(name, user, password, **kwargs)
        self._update_table_names()

    @prepare_table_name
    def append_to_table(self, table_name, table, schema=None):
        """Uploads a table with the name passed to the DataBase. If a table with same name already exists, it is
        entirely replaced by the new one. Attributes with table_names and table_schemas are updated

        :param string table_name: table name to upload to in DataBase
        :param pandas.DataFrame table: table to upload
        :param schema: Optional. Schema to look for the table in
        :return: None
        """

        if table_name not in self.tables:
            self.upload_table(table_name, table)

        else:
            self._update_table_schema(table_name, table)

            # add missing columns to table and commit it
            self._commit_table(table_name, self._add_missing_columns_to_table(table_name, table))

    def create_schema(self, schema_name, if_not_exists=True):
        if_not_exists = 'IF NOT EXISTS' if if_not_exists else ''
        self.execute(f'CREATE SCHEMA {if_not_exists} {schema_name};')

    @check_types(table_name=(str, tuple, list, set, dict))
    def create_table(self, table_name, schema=None):
        schema = schema or self.active_schema or 'public'

        if isinstance(table_name, str):
            table_name = [table_name]

        if isinstance(table_name, dict):
            # todo add variables definition option
            raise NotImplementedError()

        else:
            names = list(map(partial(join_schema_to_table_name, schema=schema), table_name))
            sqls = tuple(f"""CREATE TABLE {name} ();""" for name in names)

        for sql in sqls:
            self.execute(sql)

        self._update_table_names()

    @prepare_table_name
    def drop_primary_key(self, table_name, schema=None):
        """Drops current primary key

        :param string table_name: table name to drop primary key from
        :param schema: Optional. Schema to look for the table in
        :return: None
        """

        constraints = self.get_constraints(table_name, contype='p')['conname'].to_list()
        if constraints:
            sql = f'ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {constraints[0]};'
            self.execute(sql)

    def drop_schema(self, schema_name, cascade=False, if_exists=True):
        if_exists = "IF EXISTS" if if_exists else ''
        cascade = 'CASCADE' if cascade else ''

        if isinstance(schema_name, (tuple, list, set)):
            drop_schema = self.active_schema in schema_name
            schema_name = ','.join(schema_name)
        else:
            drop_schema = self.active_schema == schema_name

        try:
            self.execute(f'DROP SCHEMA {if_exists} {schema_name} {cascade};')

        except Exception as e:
            raise e

        else:
            if drop_schema:
                self.set_active_schema(None)

    @check_types(table_name=(str, tuple, list, set))
    def drop_table(self, table_name, schema=None, cascade=False, if_exists=True):
        if_exists = "IF EXISTS" if if_exists else ''
        cascade = 'CASCADE' if cascade else ''

        if isinstance(table_name, str):
            table_name = [table_name]

        schema = schema or self.active_schema or 'public'
        table_name = ','.join(map(partial(join_schema_to_table_name, schema=schema), table_name))

        self.execute(f'DROP TABLE {if_exists} {table_name} {cascade};')
        self._update_table_names()

    @check_options(contype=('all', 'p'))
    @prepare_table_name
    def get_constraints(self, table_name, schema=None, contype='all'):
        schema, table_name = table_name.split('.')

        sql = f"SELECT con.* FROM pg_catalog.pg_constraint con " \
              f"INNER JOIN pg_catalog.pg_class rel ON rel.oid = con.conrelid " \
              f"INNER JOIN pg_catalog.pg_namespace nsp ON nsp.oid = connamespace " \
              f"WHERE nsp.nspname = '{schema}' AND rel.relname = '{table_name}';"
        constraints = self.query(sql)

        if contype == 'all':
            return constraints
        elif contype in ('p', 'primary'):
            return constraints.loc[constraints['contype'] == 'p']

    @prepare_table_name
    def get_primary_key_columns(self, table_name, schema=None):
        key = self.get_constraints(table_name, schema, 'p')['conkey']
        cols = self.get_table_schema(table_name, schema)['column_name']
        cols.index += 1
        return cols.loc[key.to_list()[0]].to_list() if not key.empty else []

    def get_schemas(self):
        """Finds available schemas in the database"""
        sql = "SELECT schema_name FROM information_schema.schemata " \
              "WHERE schema_name !~ 'pg_catalog|information_schema|pg_toast'"
        return list(self.query(sql)['schema_name'])

    @prepare_table_name
    def get_table_schema(self, table_name, schema=None):
        schema, name = table_name.split('.')

        sql = f"SELECT column_name, data_type FROM information_schema.columns " \
              f"where table_name = '{name}' and table_schema = '{schema}'; "
        return self.query(sql)

    @check_types(table_name=str, limit=(int, NoneType))
    @prepare_table_name
    def read_table(self, table_name, limit=None, schema=None):
        """Extracts a table from the database with table_name. If limit is provided it will only extract a
        specific amount of rows from the top of the database

        :param table_name: name of the table to extract
        :param limit: amount of rows to extract
        :param schema: If the table to access is not in the active schema
        :return: pandas DataFrame with the query result
        """

        return super().read_table(table_name) if limit is None else \
            self.query(f'SELECT * FROM {table_name} LIMIT {limit}')

    def set_active_schema(self, schema=None):
        """Sets the active schema in the database. Any query will be done within the active schema without need
        to specifically identify the schema on the query

        :param schema: string name of the schema to set active
        :return:
        """

        if schema is None:
            super().set_active_schema(schema)

        elif schema in self.get_schemas():
            super().set_active_schema(schema)
            self.execute(f'ALTER USER {self.user} SET search_path TO "{self.active_schema}"')
        else:
            warnings.warn(f'\nPassed schema "{schema}" does not exist in database "{self.name}" or current user '
                          f'might not have access privileges to it. Schema was not changed.'
                          f'\nCurrent schema: {self.active_schema}', stacklevel=2)

        self._update_table_names()

    @prepare_table_name
    def set_primary_key(self, table_name, id_column, schema=None):
        """Adds a primary key to the table

        :param table_name: table to add the primary key to
        :param id_column: column to be defined as primary key. Column must exist in table, otherwise it will
        raise an error
        :param schema: Optional. Schema to look for the table in
        :return: None
        """

        # drop any existing primary key
        self.drop_primary_key(table_name)

        # if more than one column is passed join them as a single string
        if isinstance(id_column, (list, tuple, set)):
            id_column = ', '.join(id_column)

        sql = f'ALTER TABLE {table_name} ADD PRIMARY KEY ({id_column}); '
        self.execute(sql)

    @prepare_table_name
    def upload_table(self, table_name, table, schema=None):
        # drop table if exists, create the new table schema, upload the table
        self.drop_table(table_name, if_exists=True)
        self._update_table_schema(table_name, table)
        self._commit_table(table_name, table)

    @check_types(id_column_pkey=(NoneType, str, list, tuple, set))
    @prepare_table_name
    def upsert(self, table_name, table, id_column_pkey=None, schema=None):
        """If table already exists in DataBase, appends new values and on conflict on values in the primary key value,
        replaces the existing values in DataBase with the new ones. Else, creates the table.

        :param table_name: table name to upload
        :param table: table to upload
        :param id_column_pkey: column name to be used as primary key
        :return: None
        """

        if table_name not in self.tables:
            self.upload_table(table_name, table)

        else:
            table_columns = table.columns.str.lower()

            # checks on id_column_pkey
            old_constraints = self.get_constraints(table_name, contype='p')['conname'].to_list()
            if id_column_pkey is not None:
                _check = (id_column_pkey not in table_columns) if isinstance(id_column_pkey, str) else \
                    any(col not in table_columns for col in id_column_pkey)
                if _check:
                    raise ValueError('Primary key specified is not in table.')

            elif (id_column_pkey is None) and (not old_constraints):
                raise KeyError('No primary key was passed and none is set. '
                               'A primary key must exist to allow an upsert.')

            # update table schema
            self._update_table_schema(table_name, table)

            # upsert
            with self._temporary_primary_key(id_column_pkey, table_name) as new_key:
                # Add any missing col to table to be uploaded and sort accordingly
                table = self._add_missing_columns_to_table(table_name, table)

                # Create sql query and string of values to pass
                columns = ', '.join([f'{col}' for col in table.columns])
                excluded = ', '.join(f'EXCLUDED.{col}' for col in table.columns)
                sql = f"""INSERT INTO {table_name} ({columns}) VALUES %s
                          ON CONFLICT ({', '.join(new_key)})
                          DO UPDATE SET ({columns}) = ({excluded});"""

                values = list(table.to_records(index=False))
                with self.connection_manager():
                    execute_values(self._cur, sql, values)
                    self._conn.commit()

    def _add_missing_columns_to_table(self, table_name, table):
        cur_cols = self.get_table_schema(table_name)['column_name']

        _table = table.copy()
        _table.columns = _table.columns.str.lower()
        _table[list(set(cur_cols).difference(table.columns.str.lower()))] = None
        return _table[cur_cols]

    def _commit_table(self, table_name, table):
        with self.connection_manager():
            values = io.StringIO(table.to_csv(header=False, index=False, na_rep=''))
            sql = f"COPY {table_name} FROM STDIN WITH CSV DELIMITER AS ',' NULL AS '';"

            self._cur.copy_expert(sql, values)
            self._conn.commit()
            self._update_table_names()

    def _open_connection(self):
        self._conn = psycopg2.connect(dbname=self.name, user=self.user, password=self._password,
                                      host=self.host, port=self.port)
        super()._open_connection()

    @contextmanager
    def _temporary_primary_key(self, new_key, table_name):
        old_key = self.get_primary_key_columns(table_name)
        new_key = list(new_key) if not isinstance(new_key, str) else [new_key]

        if new_key != old_key:
            self.set_primary_key(table_name, new_key)

        yield new_key

        # reset key
        if not old_key:
            self.drop_primary_key(table_name)
        elif new_key != old_key:
            try:
                self.set_primary_key(table_name, old_key)
            except psycopg2.Error as e:
                self.set_primary_key(table_name, new_key)
                print(f'{e}Could not reset old primary key ({old_key}). New primary key ({new_key}) kept.')

    def _update_table_names(self):
        sql = "SELECT DISTINCT table_schema, table_name FROM information_schema.tables " \
              "WHERE table_schema !~ 'pg_catalog|information_schema'"
        schemas_and_tables = self.query(sql)
        if self.active_schema is not None:
            filt = schemas_and_tables['table_schema'] == self.active_schema
            self.tables = list(schemas_and_tables.loc[filt, 'table_name'])
        else:
            self.tables = list(schemas_and_tables['table_schema'].str.cat(schemas_and_tables['table_name'], sep='.'))

    def _update_table_schema(self, table_name, table):
        current_schema = self.get_table_schema(table_name)

        new_schema = pd.io.sql.get_schema(table, table_name)
        new_schema = new_schema.replace('"', '').replace('INTEGER', 'bigint')

        if not current_schema.empty:
            current_schema.columns = ['var', 'type']
            current_schema.set_index('var', inplace=True)
            new_schema = _convert_table_schema_to_df(new_schema)
            new_schema = _compare_table_schema(current_schema, new_schema)

            if new_schema is None:
                return

            def formatter(name, coltype, action):
                sql = f'{action.upper()} COLUMN {name} '
                sql += coltype if action == 'add' else f'TYPE {coltype}'
                return sql

            cols_schema = ',\n '.join(starmap(formatter, new_schema.to_records()))
            new_schema = f"""ALTER TABLE {table_name} {cols_schema}; """

        self.execute(new_schema)


def _convert_table_schema_to_df(schema):
    """Converts a table schema to DataFrame format

    :param string schema: PostGreSQL schema
    :return pandas.DataFrame: DataFrame with type of each column in schema
    """

    var_types = schema.lower().replace('"', '').split('(\n')[1].rstrip('\n)').split(',')
    var_types = list(map(lambda x: x.split(' ')[-2:], var_types))
    return pd.DataFrame.from_records(var_types, columns=['var', 'type']).set_index('var')


def _compare_table_schema(current_schema, new_schema):
    """Compares two PostGreSQL schemas. If there are changes from current_schema to new_schema, updates the types of
    columns of the current schema to be able to contain the types of the new_schema.
    Scale implemented: text > real > bigint > integer

    :param pandas.DataFrame current_schema: current PostGreSQL table schema
    :param pandas.DataFrame new_schema: new PostGreSQL table schema
    :return: None or DataFrame with changes in schema to upload
    """

    casting_order = {'text': 1, 'real': 2, 'bigint': 3, 'integer': 4}
    reverse_order = {v: k for k, v in casting_order.items()}
    type_map = {0: 'add', 1: 'alter'}

    if new_schema.equals(current_schema):
        return None
    else:
        _temp = pd.concat([current_schema, new_schema], axis=1)
        _temp.columns = ['cur', 'new']

        _temp = _temp.replace(casting_order)
        _temp = _temp[(_temp['cur'] > _temp['new']) | (_temp['cur'].isnull())]

        if _temp.empty:
            return None
        else:
            _temp['type'] = (~_temp['cur'].isnull()).astype(int).map(type_map)
            _temp['result'] = _temp[['cur', 'new']].min(axis=1).astype(int).replace(reverse_order)
            return _temp[['result', 'type']]
