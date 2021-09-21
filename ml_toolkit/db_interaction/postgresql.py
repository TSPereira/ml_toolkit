# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes & functions related to postgreSQL databases
"""

from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from typing import Optional, Union, Tuple, List, Any, Sequence

import numpy as np
import pandas as pd
import pandas.io.sql
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extensions import register_adapter, AsIs

from ..utils.log_utl import monitor, create_logger
from ..utils.os_utl import check_types, check_options, NoneType

from ._core import BaseActorSQL, join_schema_to_table_name, parse_schema_table


register_adapter(np.int64, AsIs)
FLAVOR = 'PostGreSQL'
LOGGER = create_logger(__name__, on_conflict='ignore', fmt=f'[%(asctime)s | {FLAVOR} | %(levelname)s] %(message)s',
                       logger_level=20)
MONITOR_LOGGER = LOGGER.getChild('monitor')
MONITOR_LOGGER.setLevel(30)

# =========================================================================================================
# ================================ 1. OTHER

# To retrieve data types in PostgreSQL
"""SELECT typname, typelem FROM pg_type
   WHERE typname LIKE '_int%';"""

# PostgreSQL type equivalent with pandas
TYPE_ALIASES = {
    'object': 'text',
    'string': 'text',
    'int64': 'bigint',
    'int32': 'integer',
    'int16': 'integer',
    'int8': 'smallint',
    'float': 'float',
    'float64': 'real',
    'float32': 'real',
    'float16': 'real',
    'float8': 'real',
    'bool': 'boolean',
    'datetime64[ns]': 'timestamp',
}

# GLOBAL VARIABLES BELOW MAYBE CAN BE DELETED
TYPE_ALIASES_REVERSED = {
    16: 'bool',
    21: 'float32',  # int16 but do not support NA so then pd.to_numeric
    20: 'float64',  # int64 but do not support NA so then pd.to_numeric
    23: 'float32',  # int32 but do not support NA so then pd.to_numeric
    25: 'object',
    700: 'float32',
    701: 'float32',
    1043: 'object',
    1114: 'object',  # datetime64 so then parse by pd.read_csv
    1700: 'float64'
}

DATES_OID = [1114]
DATE_PARSER = pd.to_datetime


# =========================================================================================================
# ================================ 2. CLASSES
# noinspection PyUnusedLocal,PyTypeChecker
class PostgreSQLManager(BaseActorSQL):
    """This class provides methods to interact with a PostgreSQL Database

    Args:
        user: User identification in the database
        password: User password to the database
        db_name: Name of the database to connect to
        host: Address to the database
        port: Port to connect through
        kwargs: Additional keyword arguments to pass to BaseActor class ("schema")

    Examples:
        >>> db = PostgreSQLManager('user', 'password', 'test', 'localhost', '5432', 'some_schema')
        [2021-05-20 12:59:11 | PostGreSQL | INFO] Connection opened to test
        [2021-05-20 12:59:12 | PostGreSQL | INFO] Active schema changed to some_schema.

    """
    _flavor = FLAVOR
    _unsafe_symbols = (';', '--')  # define symbols to be checked on arguments that can't be parsed by sql components.
    _cursor_manager_exceptions = (Exception, KeyboardInterrupt, psycopg2.Error)

    def __init__(self, user: str, password: str, host: str, port: Union[str, int] = '5432',
                 database: Optional[str] = None, **kwargs) -> None:
        self.host, self.port = host, port
        super().__init__(database, user, password, logger=LOGGER, **kwargs)
        self.refresh()

    @parse_schema_table
    def analyse(self, table_name: str, schema: Optional[str] = None) -> None:
        """Performs an analysis of a table. This computes statistics such as approximate count of rows

        Args:
            table_name: Table to be analysed
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """
        stmt = "ANALYZE {schema}.{table_name}"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        self.execute(sql.SQL(stmt).format(**params), log=f'Table {schema}.{table_name} analysed.')

    def execute(self, stmt: str, params: Optional[Tuple[Any]] = None, *, return_cursor_metadata: bool = False,
                log: Optional[str] = None) -> Optional[Union[bool, str, psycopg2.extensions.cursor]]:
        return super().execute(stmt, params, return_cursor_metadata=return_cursor_metadata, log=log)

    def get_sql(self, stmt: str, params: Optional[Tuple[Any]] = None, **kwargs) -> str:
        """Generates the sql statement with the params passed. This can be used to simulate the statements before
        sending them.

        Args:
            stmt: Sql statement to apply params to
            params: Params to insert in the sql statement
            kwargs: For compatibility

        Returns:
            String: Sql statement with params applied to
        """
        with self.cursor_manager() as cur:
            return cur.mogrify(stmt, params).decode()

    @check_types(pid=(int, list, tuple))
    def kill_process(self, pid: Union[int, List[int], Tuple[int]], *, force: bool = False) -> bool:
        """Terminate a process in the database.

        Args:
            pid: Id of the process to terminate
            force: whether to "cancel" the process softly or "terminate" forcefully.

        Returns:
            Boolean: True if successful
        """

        if isinstance(pid, (list, tuple)):
            for pid_ in pid:
                self.kill_process(pid_, force=force)

        elif isinstance(pid, int):
            return self.execute(f"SELECT pg_{'terminate' if force else 'cancel'}_backend({pid});",
                                log=f'Process {pid} was terminated successfully.')

    @check_options(state=('active', 'idle', None))
    def get_transactions(self, state: Optional[str] = None) -> pd.DataFrame:
        """List transactions in the database.

        Args:
            state: Filter transactions listed by their "state". Allowed: ('active', 'idle', None).

        Returns:
            pd.DataFrame: Table with contents of pg_stat_activity according to the filter applied.
        """
        stmt = "SELECT * FROM pg_stat_activity"
        params = dict()

        if state is not None:
            stmt += " WHERE state = {state}"
            params['state'] = sql.Literal(state)

        return self.query(sql.SQL(stmt).format(**params))

    def refresh(self) -> None:
        """Refresh the names of the existing tables.

        Returns:
            None
        """
        stmt = "SELECT DISTINCT table_schema, table_name FROM information_schema.tables " \
               "WHERE table_schema !~ 'pg_catalog|information_schema'"

        schemas_and_tables = self.query(stmt)
        self._tables = list(schemas_and_tables['table_schema'].str.cat(schemas_and_tables['table_name'], sep='.'))

    @check_types(columns=(str, list, dict), not_null=(bool, list))
    @parse_schema_table
    def add_columns(self, table_name: str, columns: Union[str, list, dict], not_null: Union[bool, List[bool]] = False,
                    schema: Optional[str] = None) -> None:
        """Add columns to a table

        Args:
            table_name: Table to add columns to
            columns: column_name or list of column_names or dict of {column_name: column_type} to be created
            not_null: Whether the created columns can have Null values or not. If a single boolean passed it will be
                applied to all columns. Alternatively it can be passed a list of booleans with specification per column.
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """

        # Preparation of inputs and sanity checks
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(columns, list):
            self.logger.warning('Adding columns without specifying their type.\nThe "text" format will be applied '
                                'which can lead to low performance.')
            columns = dict(zip(columns, ['text']*len(columns)))

        if isinstance(not_null, bool):
            not_null = [not_null]*len(columns)
        elif isinstance(not_null, list):
            assert len(not_null) == len(columns), "If explicitly passing 'NOT NULL' entries for each column, then " \
                                                  "'not null' list needs to be the same size of columns."

        # Preparation of statement and parameters
        stmt = "ALTER TABLE {schema}.{table_name}"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        for i, ((c, t), nn) in enumerate(zip(columns.items(), not_null)):
            self._check_safety(t, 'types')
            stmt += f" ADD COLUMN {{c{i}}} {TYPE_ALIASES.get(t.lower(), t)}"
            params[f'c{i}'] = sql.Identifier(c)

            if nn:
                stmt += " NOT NULL"

            stmt += ','
        else:
            stmt = stmt[:-1]

        self.execute(sql.SQL(stmt).format(**params),
                     log=f'Columns {tuple(columns)} added successfully to {schema}.{table_name}.')

    @monitor(mode='time', logger=MONITOR_LOGGER)
    @check_types(columns=dict)
    @parse_schema_table
    def alter_columns(self, table_name: str, columns: dict, using: Optional[Union[str, List[str]]] = None,
                      schema: Optional[str] = None) -> None:
        """Alter column types in a table

        Args:
            table_name: Table to alter columns on
            columns: dictionary of format {column_name: column_type}.
            using: Optional. expression or list of expressions to use to convert. If list, needs to be the same size
                as "columns". Expressions cannot contain "unsafe_symbols" such as ";" or "--" to avoid injection.
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """
        # Preparation of inputs and sanity checks
        if isinstance(using, str) or using is None:
            using = [using]*len(columns)

        assert len(using) == len(columns), 'Size of "using" needs to be the same as "columns".'

        # Preparation of statement and parameters
        stmt = "ALTER TABLE {schema}.{table_name}"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))

        for i, ((c, t), u) in enumerate(zip(columns.items(), using)):
            self._check_safety(t, 'types')
            stmt += f" ALTER COLUMN {{c{i}}} TYPE {t}"
            params[f'c{i}'] = sql.Identifier(c)

            if u is not None:
                self._check_safety(u, 'using')
                stmt += f" USING {{c{i}}}::{u}"

            stmt += ','
        else:
            stmt = stmt[:-1]

        self.execute(sql.SQL(stmt).format(**params),
                     log=f'Columns {columns} of table {schema}.{table_name} changed successfully.')

    @check_types(columns=(str, list, tuple), cascade=(bool, list))
    @parse_schema_table
    def drop_columns(self, table_name: str, columns: Union[str, list, tuple], cascade: Union[bool, List[bool]] = False,
                     if_exists: bool = True, schema: Optional[str] = None) -> None:
        """Drops columns from table

        Args:
            table_name: Table to drop columns from
            columns: column_name or list of column_names to drop
            cascade: boolean or list of booleans to control whether dependent objects are dropped or an error is raised.
                If a list is passed it needs to be of the same length of "columns". If a single boolean is passed it
                will be applied to all columns
            if_exists: Whether to raise an error or not if the column doesn't exist
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(cascade, bool):
            cascade = [cascade]*len(columns)
        elif isinstance(cascade, list):
            assert len(cascade) == len(columns), "If explicitly passing 'cascade' entries for each column, then " \
                                                  "'cascade' list needs to be the same size of columns."

        stmt = "ALTER TABLE {schema}.{table_name}"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        for i, (col, cas) in enumerate(zip(columns, cascade)):
            stmt += " DROP COLUMN" + " IF EXISTS"*if_exists + f" {{c{i}}}" + " CASCADE"*cas + ','
            params[f'c{i}'] = sql.Identifier(col)
        else:
            stmt = stmt[:-1]

        self.execute(sql.SQL(stmt).format(**params),
                     log=f'Dropped columns {tuple(columns)} from "{schema}.{table_name}" successfully.')

    @parse_schema_table
    def rename_column(self, table_name: str, old_column_name: str, new_column_name: str,
                      schema: Optional[str] = None) -> None:
        """Rename a column in a table

        Args:
            table_name: Table that contains the column to be renamed
            old_column_name: Original column name
            new_column_name: New column name
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """
        stmt = "ALTER TABLE {schema}.{table_name} RENAME {old_column_name} TO {new_column_name};"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name),
                      old_column_name=sql.Identifier(old_column_name), new_column_name=sql.Identifier(new_column_name))
        self.execute(sql.SQL(stmt).format(**params), log=f'Column "{old_column_name}" of table "{schema}.{table_name}" '
                                                         f'renamed to "{new_column_name}" successfully.')

    @monitor(mode='time', logger=MONITOR_LOGGER)
    @check_types(columns=(str, list), unique=bool)
    @check_options(on_conflict=('raise', 'drop'))
    @parse_schema_table
    def create_index(self, table_name: str, columns: Union[str, list], index_name: Optional[str] = None,
                     unique: bool = False, on_conflict: str = 'raise', schema: Optional[str] = None) -> None:
        """Creates an index on a table on the given columns

        Args:
            table_name: table to create the index one
            columns: which columns to use for the index
            index_name: Optional. name to use for index. If none is passed the name will be generated automatically
            unique: whether the index should have unique entries
            on_conflict: Whether to raise an error or drop the index if an index with the same name exists in the schema
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None

        Examples:
            >>> db = PostgreSQLManager(...)
            >>> db.create_index('test_table', columns=['a', 'b'], unique=True)
            [2021-05-20 12:59:11 | PostGreSQL | INFO] Index idx_public_test_table_a_b successfully created.
        """

        if isinstance(columns, str):
            columns = [columns]

        # Sorting to ensure consistency in the index whatever the order of given columns
        # define index name
        columns = sorted([c.replace(' ', '_') for c in columns])
        index_name = index_name or f"{schema}_{table_name}_{'_'.join(columns)}_idx"

        # check whether the index exists and if should be dropped or not
        if not self.get_index(index_name, schema).empty and on_conflict == 'raise':
            raise IndexError(f'Index with name {schema}.{index_name} already exists. Check if you need to create '
                             f'this index.\nIf you wish to drop it and recreate pass on_conflict=="drop".')

        # Drop if already exists with same name and creates
        self.drop_index(f'{schema}.{index_name}')
        stmt = "CREATE" + " UNIQUE"*unique + " INDEX {index_name} ON {schema}.{table_name} ({columns});"
        params = dict(index_name=sql.Identifier(index_name), table_name=sql.Identifier(table_name),
                      schema=sql.Identifier(schema), columns=sql.SQL(",").join(map(sql.Identifier, columns)))

        self.execute(sql.SQL(stmt).format(**params), log=f'Index {index_name} successfully created.')

    def drop_index(self, index_name: str, *, cascade: bool = False, return_query: bool = False) -> Optional[str]:
        """Drops an existing index by its name.

        Args:
            index_name: index name to drop
            cascade: Automatically drop objects that depend on the index.
            return_query: whether to return the original query used to create the index being dropped.

        Returns:
            Optional[String]: Query used to create index being dropped or empty string

        Examples:
            >>> db = PostgreSQLManager(...)
            >>> db.drop_index('idx_public_test_index_col_a', return_query=True)
            'CREATE INDEX idx_public_test_index_col_a ON public.test_index (col_a)'
        """

        # parse idx_name (can be passed with schema)
        schema = None
        if '.' in index_name:
            schema, index_name = index_name.split('.', 1)

        # check if the idx_exists
        idx = self.get_index(index_name, schema)
        if idx.empty:
            return

        # if there is no schema passed get it from the table
        if schema is None:
            if idx.shape[0] > 1:
                raise IndexError(f'There are multiple indexes with name {index_name}. Please pass explicitly the '
                                 f'schema in the index_name as "<schema>.<index_name>".')
            schema = idx.iloc[0]['schemaname']

        stmt = "DROP INDEX IF EXISTS {schema}.{index_name}" + " CASCADE" * cascade
        self.execute(sql.SQL(stmt).format(index_name=sql.SQL(index_name), schema=sql.Identifier(schema)),
                     log=f'Index {index_name} successfully dropped.')

        if return_query:
            return idx[idx['schemaname'] == schema]['indexdef'].iloc[0]

    def get_index(self, idx_name: str, schema: Optional[str] = None) -> pd.DataFrame:
        """Retrieve information for a specific index

        Args:
            idx_name: Name of the index to retrieve info for
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            pd.DataFrame: Table with information. Typically it should have a single row
        """
        idxs = self.get_indexes(schema=schema)
        return idxs[idxs.indexname == idx_name]

    def get_indexes(self, table_name: Optional[str] = None, schema: Optional[str] = None) -> pd.DataFrame:
        """List all indexes. Might be filtered by schema/table

        Args:
            table_name: name of the table if filtering by table
            schema: name of the schema if filtering by schema

        Returns:
            pd.DataFrame: table with the indexes information filtered according to the parameters passed
        """
        if table_name is not None and '.' in table_name:
            if schema is not None:
                self.logger.warning('Schema passed in "table_name". The "schema" argument will be ignored.')
            schema, table_name = table_name.split('.')

        stmt = "SELECT * FROM pg_indexes WHERE schemaname != 'pg_catalog'"
        params = dict()

        if schema is not None:
            stmt += " AND schemaname = {schema}"
            params['schema'] = sql.Literal(schema)

        if table_name is not None:
            stmt += " AND tablename = {table_name}"
            params['table_name'] = sql.Literal(table_name)

        return self.query(sql.SQL(stmt).format(**params))

    def get_indexes_columns(self, index_name: Optional[Union[str, list]] = None, table_name: Optional[str] = None,
                            schema: Optional[str] = None) -> pd.Series:
        """Get columns for indexes in a table, schema or specific index_name

        Args:
            index_name: Specific index_name string or list of index_names
            table_name: Tables to search the indexes on
            schema: Schema to search the indexes on

        Returns:
            pd.Series: index=index_name, values=lists of columns
        """
        idxs = self.get_indexes(table_name, schema=schema)

        if index_name is not None:
            if isinstance(index_name, str):
                index_name = [index_name]
            idxs = idxs[idxs['indexname'].isin(index_name)]

        idxs['cols'] = idxs['indexdef'].str.extract(r'.*\((.*)\)')
        idxs['cols'] = idxs['cols'].str.split(', ').apply(sorted)
        idxs.set_index('indexname', inplace=True)
        return idxs['cols']

    def create_schema(self, schema_name: str, if_not_exists: bool = True) -> None:
        """Creates a new schema in the database

        Args:
            schema_name: Name for the new schema
            if_not_exists: If False will raise an error if the a schema with the same name already exists.

        Returns:
            None

        Examples:
            >>> db = PostgreSQLManager(...)
            >>> db.create_schema('test_schema')
            [2021-05-20 12:59:11 | PostGreSQL | INFO] Schema test_schema successfully created in <name> database.
        """

        stmt = "CREATE SCHEMA" + " IF NOT EXISTS"*if_not_exists + " {schema_name}"
        self.execute(sql.SQL(stmt).format(schema_name=sql.Identifier(schema_name)),
                     log=f'Schema {schema_name} successfully created in {self.name} database.')

    def drop_schema(self, schema_name: Union[str, list], cascade: bool = False, if_exists: bool = True) -> None:
        """Drops an entire schema. If "cascade" is True it will drop any table inside of that schema. Allows multiple
        schemas to be dropped at same time by passing a list of schema names. If the active_schema is dropped it will
        reset the instance "active_schema" to "public".

        Args:
            schema_name: name of the schema to drop
            cascade: Whether to drop everything inside of the schema or to raise an error if any table is still in the
                schema.
            if_exists: Whether to ignore or raise an error if the schema doesn't exist.

        Returns:
            None
        """

        if isinstance(schema_name, (tuple, list, set)):
            drop_schema = self.active_schema in schema_name
            _schema_name = sql.SQL(",").join(map(sql.Identifier, schema_name))
        else:
            drop_schema = self.active_schema == schema_name
            _schema_name = sql.Identifier(schema_name)

        stmt = "DROP SCHEMA" + " IF EXISTS"*if_exists + " {schema_name}" + " CASCADE"*cascade
        if self.execute(sql.SQL(stmt).format(schema_name=_schema_name),
                        log=f'Schema(s) {schema_name} dropped successfully.'):

            if drop_schema:
                self.set_active_schema(None)

    def get_schemas(self) -> list:
        """Finds available schemas in the database

        Returns:
            list: list of schemas in database
        """
        stmt = "SELECT schema_name FROM information_schema.schemata " \
               "WHERE schema_name !~ 'pg_catalog|information_schema|pg_toast'"
        return list(self.query(stmt)['schema_name'])

    def set_active_schema(self, schema: Optional[str] = None) -> None:
        """Sets the active schema in the database. Any query will be done within the active schema without need
        to specifically identify the schema on the query

        Args:
            schema: If None is passed it will set the active_schema to "public"

        Returns:
            None
        """

        schemas = self.get_schemas()
        if (schema is not None) and (schema not in schemas):
            self.logger.warning(f'Passed schema "{schema}" does not exist in database "{self.name}" or current user '
                                f'might not have access privileges to it.'
                                f'\nSchema was not changed. Current schema: {self.active_schema}')
        else:
            super().set_active_schema(schema)
            self.refresh()

            if schema in schemas:
                stmt = 'ALTER USER {user} SET search_path TO {active_schema}'
                params = dict(user=sql.Identifier(self.user), active_schema=sql.Identifier(self.active_schema))
                self.execute(sql.SQL(stmt).format(**params))

    @parse_schema_table
    def drop_primary_key(self, table_name: str, schema: Optional[str] = None) -> None:
        """Drops current primary key

        Args:
            table_name: table name to drop primary key from
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """

        constraints = self.get_constraints(table_name, contype='p', schema=schema)['conname']
        if not constraints.empty:
            stmt = "ALTER TABLE {schema}.{table_name} DROP CONSTRAINT IF EXISTS {constraint}"
            params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name),
                          constraint=sql.SQL(constraints[0]))
            self.execute(sql.SQL(stmt).format(**params), log=f'Primary key {constraints[0]} dropped.')

    @parse_schema_table
    def get_primary_key(self, table_name: str, schema: Optional[str] = None) -> pd.Series:
        """Retrieve all info about the primary key of a table

        Args:
            table_name: Table to get the primary key from
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:

        """
        return self.get_constraints(table_name, 'p', schema=schema)

    @parse_schema_table
    def get_primary_key_columns(self, table_name: str, idx: bool = False, schema: Optional[str] = None) -> list:
        """Find the columns of the primary_key constraint for a given table

        Args:
            table_name: Name to get primary key from
            idx: Whether to return the columns as names or indexes. Default: False (returns names).
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            List: list of columns that are the primary keys
        """
        key = self.get_primary_key(table_name, schema=schema)['conkey']
        if key.empty:
            return []

        key = key.iloc[0]
        if not idx:
            cols = self.get_columns(table_name, schema)
            cols.insert(0, '_')
            key = itemgetter(*key)(cols)
            key = [key] if not isinstance(key, tuple) else list(key)

        return sorted(key)

    @monitor(mode='time', logger=MONITOR_LOGGER, log_level='debug')
    @check_options(on_conflict=('raise', 'drop'))
    @parse_schema_table
    def set_primary_key(self, table_name: str, id_column: Union[str, list, tuple], on_conflict='raise',
                        schema: Optional[str] = None) -> Optional[bool]:
        """Adds a primary key to the table.

        Args:
            table_name: table to add the primary key to
            id_column: column(s) to be defined as primary key. Column must exist in table, otherwise it will raise
                an error
            on_conflict: What to do if already exists a primary key set for this table. Options: 'raise', 'drop'.
                Default: 'raise'
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            If True the set of the key was completed successfully
        """

        # if more than one column is passed join them as a single string
        if not isinstance(id_column, (list, tuple, set)):
            id_column = [id_column]

        stmt = "ALTER TABLE {schema}.{table_name} ADD PRIMARY KEY ({id_column})"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name),
                      id_column=sql.SQL(',').join(map(sql.Identifier, id_column)))

        if on_conflict == 'drop':
            self.drop_primary_key(table_name, schema)

        return self.execute(sql.SQL(stmt).format(**params), log=f'Primary key set on {tuple(id_column)} successfully.')

    @monitor(logger=MONITOR_LOGGER)
    @parse_schema_table
    def append_to_table(self, table_name: str, values: Union[pd.DataFrame, Sequence],
                        columns: Optional[Union[list, dict]] = None, on_new_columns: str = 'raise',
                        schema: Optional[str] = None) -> None:
        """Uploads a table with the name passed to the DataBase. If the table doesn't exist yet, will create a new one
        else it will append any missing columns to the dataframe passed (to ensure consistency) with null values and
        add it. New columns that don't exist on the destination table will raise a error.

        Args:
            table_name: table name to upload to in DataBase
            values: table to upload
            columns: list, dict or None. Definition of new columns (or new table) requires a dictionary of format
                            {<col_name>: <col_type>} except if 'values' is a DataFrame.
            on_new_columns: What to do if new columns are in the DataFrame passed. Options: ('raise', 'ignore', 'add').
                "raise": raises an error
                "ignore": drop the additional columns from the dataframe before uploading it. If values is not a
                            DataFrame it will raise an error instead.
                "add": create columns in the database table with inferred type before uploading the data.
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None

        Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> db = PostgreSQLManager(...)
            >>> db.append_to_table('test_table', df)
        """

        self.refresh()
        full_tablename = f'{schema}.{table_name}'

        if full_tablename not in self._tables:
            if not isinstance(values, pd.DataFrame) and isinstance(columns, (list, NoneType)):
                raise TypeError(f'Table {full_tablename} does not exist. To create the table with "append_to_table" '
                                f'method, "columns" must be a dictionary defining the types of each column.')
            self.upload_table(table_name, values, columns, schema=schema)

        else:
            # if the table already exists, update its schema and add any missing columns before appending the new table
            values, columns = self._update_table_schema(full_tablename, values, columns, on_new_columns)
            if self._commit_table(table_name, values, columns, update_table_names=False, schema=schema):
                self.logger.info(f'Data appended successfully to table {full_tablename}.')

    @monitor(logger=MONITOR_LOGGER)
    @parse_schema_table
    def copy_table(self, table_name: str, new_table_name: str, columns: Optional[list] = None,
                   where: Optional[str] = None, structure_only: bool = False, schema: Optional[str] = None,
                   destination_schema: Optional[str] = None) -> None:
        """Copies one table to another. The copy can include all of the data, just a selection of the data or
        structure only.

        Args:
            table_name: Name of the table to copy
            new_table_name: Name of the new table
            columns: Which columns to include on the copy
            where: Sql statement to select specific rows to be copied. This statement cannot use any "unsafe_symbol"
                such as ";" or "--".
            structure_only: Whether to copy only the structure of the table (without any data)
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema
            destination_schema: Optional. Schema to copy the table to. If 'None' uses the instance active_schema

        Returns:
            None
        """

        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        stmt = "CREATE TABLE {new_schema}.{new_table_name} AS"

        # parse and add new schema and table name
        destination_schema = destination_schema or self.active_schema
        new_schema, new_table_name = join_schema_to_table_name(new_table_name, destination_schema).split('.', 1)
        params['new_schema'], params['new_table_name'] = map(sql.Identifier, (new_schema, new_table_name))

        if (columns is None) and (where is None):
            stmt += " TABLE {schema}.{table_name}"

        else:
            stmt += " SELECT {columns} FROM {schema}.{table_name}"
            params['columns'] = sql.SQL('*') if columns is None else sql.SQL(",").join(map(sql.Identifier, columns))

            if where is not None:
                self._check_safety(where, 'where')
                stmt += f" WHERE {where}"

        if structure_only:
            stmt += " WITH NO DATA"

        log = f'Table {schema}.{table_name} copied successfully to {new_schema}.{new_table_name}'
        if self.execute(sql.SQL(stmt).format(**params), log=log):
            self.refresh()

    @check_types(table_name=str, types=(NoneType, dict))
    @parse_schema_table
    def create_empty_table(self, table_name: str, types: Optional[dict] = None,
                           from_df: Optional[pd.DataFrame] = None, if_not_exists: bool = False,
                           schema: Optional[str] = None) -> None:
        """Creates a new empty table in the selected (or active if None passed) schema.

        Args:
            table_name: Name for the table
            types: dictionary of format {column_name: column_type}. If a DataFrame is passed in "from_df",
                "types" is used to override any inferred type from pandas
            from_df: A DataFrame to use to infer the structure of the table from
            if_not_exists: Boolean flag to raise or not an error if a table with the same name already exists
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None

        Examples:
            >>> db = PostgreSQLManager(...)
            >>> db.create_empty_table('test_table', ['a', 'b'], ['int32', 'string'])
            [2021-05-20 12:59:11 | PostGreSQL | INFO] Table test_table successfully created.
        """

        # get known types from global dict
        if types is not None:
            types = {col: TYPE_ALIASES.get(str(t).lower(), str(t)) for col, t in types.items()}

        # if a df was passed extract missing types from it
        if from_df is not None:
            types = _get_col_types_from_df_schema(pd.io.sql.get_schema(from_df, '_', dtype=types))

        # define columns as sql.Components or empty
        feat = tuple(map(lambda x: sql.SQL('{c} {t}').format(c=sql.Identifier(x[0]), t=sql.SQL(str(x[1]))),
                         types.items() if types is not None else ''))

        stmt = "CREATE TABLE" + " IF NOT EXISTS"*if_not_exists + " {schema}.{table_name} ({feat})"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name),
                      feat=sql.SQL(',').join(feat))

        if self.execute(sql.SQL(stmt).format(**params), log=f'Table {table_name} successfully created.'):
            self.refresh()

    @check_types(table_name=str, where=(str, NoneType))
    @parse_schema_table
    def delete_rows(self, table_name: str, where: Optional[str] = None, schema: Optional[str] = None) -> None:
        """Delete one or more rows based on a given condition in the database.

        Args:
            table_name: Name of the table from where to delete the row
            where: Sql statement to select specific rows to be dropped, ex: 'station_id = "test_id_to_delete"'.
                    This statement cannot use any "unsafe_symbol" such as ";" or "--".
            schema: Schema in which the table is stored

        Returns:

        """

        stmt = "DELETE FROM {schema}.{table_name}"
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))

        if where:
            self._check_safety(where, 'where')
            stmt += f" WHERE {where}"

        cur_meta = self.execute(sql.SQL(stmt).format(**params), return_cursor_metadata=True)
        if cur_meta:
            self.logger.info(f'Deleted {cur_meta.rowcount} rows from {schema}.{table_name} successfully.')

    @check_types(table_name=(str, tuple, list, set))
    def drop_table(self, table_name: Union[str, list], cascade: bool = False, if_exists: bool = True,
                   schema: Optional[str] = None) -> None:
        """Drops a table. If "cascade" is True it will drop any index or data related to that table. Allows multiple
        tables to be dropped at same time by passing a list of table names.

        Args:
            table_name: Table Name (or Names) to drop. names can be passed with or without a schema included.
            cascade: Whether to drop everything related to the table or to raise an error if any data still exists.
            if_exists: Whether to ignore or raise an error if the table doesn't exist.
            schema: Schema to use for any table without a schema explicitly provided

        Returns:
            None
        """

        def convert_tablename_to_sql_id(t):
            return sql.SQL('.').join(map(sql.Identifier, t.split('.', 1)))

        if isinstance(table_name, str):
            table_name = [table_name]

        schema = schema or self.active_schema or 'public'
        table_name = list(map(partial(join_schema_to_table_name, schema=schema), table_name))
        table_names = sql.SQL(',').join(map(convert_tablename_to_sql_id, table_name))
        stmt = "DROP TABLE" + " IF EXISTS"*if_exists + " {table_names}" + ' CASCADE'*cascade

        if self.execute(sql.SQL(stmt).format(table_names=table_names),
                        log=f'Table(s) {table_name} dropped successfully.'):
            self.refresh()

    @monitor(logger=MONITOR_LOGGER)
    @check_types(table_name=str, limit=(int, NoneType), columns=(NoneType, list, tuple, set, str))
    @check_options(sort_dir=('ASC', 'asc', 'DESC', 'desc'))
    @parse_schema_table
    def get_table(self, table_name: str, columns: Optional[Union[Sequence, str]] = None, limit: Optional[int] = None,
                  where: Optional[str] = None, order_by: Optional[Union[str, List[str]]] = None, sort_dir: str = 'ASC',
                  schema: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Extracts a table from the database with table_name. If limit is provided it will only extract a
        specific amount of rows from the top of the database

        Args:
            table_name: name of the table to extract
            columns: Sequence of columns to read from the table
            limit: amount of rows to extract
            where: Condition for the where statement (e.g 'id = 3 AND project = "open source"'). Default: None
            order_by: SQL order by statement. Default: None
            sort_dir: Direction to order by ('ASC' or 'DESC'). Default: 'ASC'
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema
            **kwargs: additional arguments to pass to pd.read_sql

        Returns:
            pd.DataFrame: Table with the result of the query
        """
        if isinstance(columns, str):
            columns = [columns]

        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        params['columns'] = sql.SQL('*') if columns is None else sql.SQL(",").join(map(sql.Identifier, columns))
        stmt = 'SELECT {columns} FROM {schema}.{table_name}'

        if where is not None:
            self._check_safety(where, 'where')
            stmt += f" WHERE {where}"

        if order_by is not None:
            stmt += " ORDER BY"

            if isinstance(order_by, str):
                order_by = [order_by]

            for i, o in enumerate(order_by):
                params[f'order_by_{i}'] = sql.Identifier(o)
                stmt += f' {{order_by_{i}}} {sort_dir.upper()},'
            else:
                stmt = stmt[:-1]

        if limit is not None:
            params['limit'] = sql.Literal(limit)
            stmt += ' LIMIT {limit}'

        return self.query(sql.SQL(stmt).format(**params), **kwargs)

    @parse_schema_table
    def rename_table(self, table_name: str, new_name: str, if_exists: bool = False,
                     schema: Optional[str] = None) -> None:
        """Renames a table

        Args:
            table_name: Table to rename
            new_name: New name to give to table
            if_exists: Whether it should raise or not an error if the table doesn't exist
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """

        new_name = new_name.split('.')[-1]
        stmt = "ALTER TABLE" + " IF EXISTS"*if_exists + " {o_schema}.{o_table} RENAME TO {n_table};"
        params = dict(o_schema=sql.Identifier(schema), o_table=sql.Identifier(table_name),
                      n_table=sql.Identifier(new_name))

        log = f'Renamed table {schema}.{table_name} to {schema}.{new_name} successfully.'
        if self.execute(sql.SQL(stmt).format(**params), log=log):
            self.refresh()

    @monitor(logger=MONITOR_LOGGER)
    @check_types(columns=(str, list), where=(str, NoneType))
    @parse_schema_table
    def update_table(self, table_name: str, columns: Union[str, list], values_or_expressions: Union[Any, Sequence],
                     where: Optional[str] = None, schema: Optional[str] = None) -> None:
        """Update a table on specific columns according to values or expressions and a where statement
            For more complex updates one should consider using upsert_table

        Args:
            table_name: Table to update
            columns: Columns to update
            values_or_expressions: values or expressions to insert. The number of values or expressions must
                be the same as the number of columns passed
            where: where statement to filter the table. Updates will only be done in rows that respect this statement
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """

        if isinstance(columns, str):
            columns = [columns]

        if isinstance(values_or_expressions, str) or not isinstance(values_or_expressions, Sequence):
            values_or_expressions = [values_or_expressions]

        # check lengths
        if len(columns) != len(values_or_expressions):
            raise IndexError(f'If "values_or_expressions" is a "Sequence" it must be of same size of "columns". '
                             f'(columns: {len(columns)} != values: {len(values_or_expressions)}.')

        elif len(columns) == 0:
            raise ValueError('At least one column must be passed.')

        # check expressions safety
        values_or_expressions = [str(v) for v in values_or_expressions]
        [self._check_safety(expr, 'expressions') for expr in values_or_expressions]

        # build stmt
        params = dict(table_name=sql.Identifier(table_name), schema=sql.Identifier(schema),
                      columns=sql.SQL(', ').join(map(sql.Identifier, columns)))
        stmt = "UPDATE {schema}.{table_name} SET "

        if len(columns) == 1:
            stmt += "{columns} = " + values_or_expressions[0]
        else:
            stmt += "({columns}) = " + f"({', '.join(values_or_expressions)})"

        if where is not None:
            self._check_safety(where, 'where')
            stmt += f" WHERE {where}"

        cur_meta = self.execute(sql.SQL(stmt).format(**params), return_cursor_metadata=True)
        if cur_meta:
            self.logger.info(f'Updated {cur_meta.rowcount} rows from {schema}.{table_name} successfully.')

    @monitor(logger=MONITOR_LOGGER)
    @check_types(columns=(NoneType, dict))
    @check_options(on_conflict=('raise', 'drop'))
    @parse_schema_table
    def upload_table(self, table_name: str, values: Union[pd.DataFrame, Sequence], columns: Optional[dict] = None,
                     on_conflict='raise', schema: Optional[str] = None) -> None:
        """Uploads a table to the database. If a table already exists with the same name in the schema it will drop it.

        Args:
            table_name: Name of the table to upload to
            values: Dataframe or Sequence with data to upload
            columns: Optional. dict of format {column_name: column_type} to define column types.
                        If table is a DataFrame missing types will be inferred
            on_conflict: Whether to raise an error or drop table if a table with same name already exists.
                Options ('raise', 'drop'). Default: 'raise'.
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """
        self._check_integrity(values, columns)

        # drop table if exists, create the new table schema, upload the table
        fulltable_name = f'{schema}.{table_name}'
        if fulltable_name in self._tables:
            if on_conflict == 'drop':
                self.drop_table(table_name, schema=schema)
            else:
                raise KeyError(f'Table {fulltable_name} already exists. If you want to replace it pass explicitly: '
                               f'on_conflict="drop"')

        if isinstance(values, pd.DataFrame):
            self.create_empty_table(table_name, columns, from_df=values, schema=schema)
            try:
                values, columns = values.to_numpy(na_value=None), values.columns.to_list()
            except TypeError:
                values, columns = values.to_numpy(), values.columns.to_list()
        else:
            self._check_integrity(values, columns or [])
            self.create_empty_table(table_name, columns, schema=schema)

        self._commit_table(table_name, values, columns, schema=schema)

    @monitor(logger=MONITOR_LOGGER)
    @check_types(id_column_pkey=(NoneType, str, list, tuple))
    @parse_schema_table
    def upsert_table(self, table_name: str, table: Union[pd.DataFrame, Sequence],
                     columns: Optional[Union[list, dict]] = None,
                     id_column_pkey: Optional[Union[str, list, tuple]] = None,
                     on_new_columns: str = 'raise', schema: Optional[str] = None) -> None:
        """Uploads/inserts a table to the database. On collisions of the id_column_pkey used it will override the lines,
        otherwise appends to table. If table doesn't exist, creates a new one.

        Args:
            table_name: Name of the table to upload to
            table: Dataframe with data to upload
            columns: list, dict or None. Definition of new columns (or new table) requires a dictionary of format
                            {<col_name>: <col_type>} except if 'values' is a DataFrame.
            id_column_pkey: list of columns to use as keys for collision comparison
            on_new_columns: What to do if new columns are in the DataFrame passed. Options: ('raise', 'ignore', 'add').
                "raise": raises an error
                "ignore": drop the additional columns from the dataframe before uploading it
                "add": create columns in the database table with inferred type before uploading the data.
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            None
        """

        def sql_join(func, iterable):
            return sql.SQL(', ').join(map(func, iterable))

        def excluded_cols_composition(col):
            return sql.SQL('EXCLUDED.{col}').format(col=sql.Identifier(col))

        self.refresh()
        fulltable_name = f'{schema}.{table_name}'

        if fulltable_name not in self._tables:
            self.upload_table(fulltable_name, table, columns)

        else:
            # checks on id_column_pkey
            if id_column_pkey is None:
                id_column_pkey = self.get_primary_key_columns(fulltable_name)
                if not id_column_pkey:
                    raise KeyError('No primary key was passed and none is set. '
                                   'A primary key must exist to allow an upsert.py.')
            else:
                id_column_pkey = [id_column_pkey] if isinstance(id_column_pkey, str) else id_column_pkey
                id_column_pkey = sorted([k.lower() for k in id_column_pkey])

            # upsert
            table, columns = self._update_table_schema(fulltable_name, table, columns, on_new_columns)
            with self._temporary_primary_key(id_column_pkey, fulltable_name) as new_key:

                # Create sql query and string of values to pass
                params = dict(pk=sql_join(sql.Identifier, new_key),
                              columns=sql_join(sql.Identifier, columns),
                              excluded=sql_join(excluded_cols_composition, columns))
                on_conflict = sql.SQL("ON CONFLICT ({pk}) DO UPDATE SET ({columns}) = ({excluded})").format(**params)

                if self._commit_table(fulltable_name, table, columns, on_conflict, update_table_names=False):
                    self.logger.info(f'Upsert into "{schema}.{table_name}" successful.')

    @parse_schema_table
    def get_columns(self, table_name: str, schema: Optional[str] = None) -> List[str]:
        """Returns the columns of a table

        Args:
            table_name: Name to get columns from
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            list: list of columns of a table
        """
        stmt = "SELECT column_name FROM information_schema.columns " \
               "WHERE table_schema = {schema} AND table_name = {table_name}"
        params = dict(schema=sql.Literal(schema), table_name=sql.Literal(table_name))
        return self.query(sql.SQL(stmt).format(**params))['column_name'].to_list()

    @check_options(contype=('all', 'p', 'primary', 'u', 'unique'))
    @parse_schema_table
    def get_constraints(self, table_name: str, contype: str = 'all', schema: Optional[str] = None) -> pd.DataFrame:
        """Retrieve the constraints associated to a table.

        Args:
            table_name: table to get constraints for
            contype: type of constraints that can be queries. Options: "all", "p"(or "primary"). Default: "all"
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            pd.DataFrame: Table with all info on the constraints selected.
        """

        stmt = "SELECT con.* FROM pg_catalog.pg_constraint con " \
               "INNER JOIN pg_catalog.pg_class rel ON rel.oid = con.conrelid " \
               "INNER JOIN pg_catalog.pg_namespace nsp ON nsp.oid = connamespace " \
               "WHERE nsp.nspname = {schema} AND rel.relname = {table_name}"
        params = dict(schema=sql.Literal(schema), table_name=sql.Literal(table_name))

        if contype in ('p', 'primary', 'u', 'unique'):
            params['constraint'] = sql.Literal(contype[0])
            stmt += " AND con.contype = {constraint}"

        return self.query(sql.SQL(stmt).format(**params))

    @parse_schema_table
    def get_dtypes(self, table_name: str, columns: Optional[Union[str, list]] = None,
                   schema: Optional[str] = None) -> pd.Series:
        """Get the types of each column in a table

        Args:
            table_name: table to get types for
            columns: str or list of strings identifying columns to be retrieved. Will only return existing columns.
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            pd.Series: table containing dtypes for each column in table and column names as the index
        """

        params = dict(schema=sql.Literal(schema), table_name=sql.Literal(table_name))
        stmt = """SELECT column_name,
                    CASE
                        WHEN domain_name IS NOT NULL THEN domain_name
                        WHEN data_type='character varying' THEN 'varchar('||character_maximum_length||')'
                        WHEN data_type='numeric' THEN 'numeric('||numeric_precision||','||numeric_scale||')'
                        ELSE data_type
                    END AS data_type
                    FROM information_schema.columns
                    WHERE table_schema = {schema} AND table_name = {table_name}"""

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]

            stmt += " AND column_name in {columns}"
            params['columns'] = sql.Literal(tuple(columns))

        return self.query(sql.SQL(stmt).format(**params)).set_index('column_name')['data_type']

    @parse_schema_table
    def get_na(self, table_name: str, columns: Optional[Union[str, list]] = None, relative: bool = False,
               schema: Optional[str] = None) -> pd.Series:
        """Get number of missing values in each column of the table

        Args:
            table_name: table to get number of uniques for
            columns: str or list of strings identifying columns to be retrieved. Will only return existing columns.
            relative: Whether to return % of missing values or absolute values
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            pd.Series: pandas Series with count of missing values as values and column names as index
        """
        # manage columns to present
        existing_columns = self.get_columns(table_name, schema)

        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = existing_columns

        columns = [col for col in columns if col in existing_columns]

        # Prepare stmt
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        stmt = "SELECT "
        for i, column in enumerate(columns):
            stmt += f'(COUNT(*) - COUNT({{c{i}}}))' + '/ COUNT(*)::real'*relative + f' AS {column}, '
            params[f'c{i}'] = sql.Identifier(column)
        else:
            stmt = stmt[:-2] + " FROM {schema}.{table_name}"

        return self.query(sql.SQL(stmt).format(**params)).iloc[0].T

    @parse_schema_table
    def get_nunique(self, table_name: str, columns: Optional[Union[str, list]] = None, count_null: bool = False,
                    schema: Optional[str] = None) -> pd.Series:
        """Get number of unique values in each column of the table

        Args:
            table_name: table to get number of uniques for
            columns: str or list of strings identifying columns to be retrieved. Will only return existing columns.
            count_null: Whether to count an existing null value as a distinct value
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            pd.Series: pandas Series with count of uniques as values and column names as index
        """

        # manage columns to present
        existing_columns = self.get_columns(table_name, schema)

        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = existing_columns

        columns = [col for col in columns if col in existing_columns]

        # build query
        stmt = ""
        for i, column in enumerate(columns):
            stmt += f"SELECT COUNT({'*' if count_null else column}) FROM (" \
                        f"SELECT DISTINCT {column} AS {column} " \
                        "FROM {schema}.{table_name})" + f" AS t{i} " \
                        "UNION ALL\n"
        else:
            stmt = stmt[:-11]

        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        df = self.query(sql.SQL(stmt).format(**params))
        df.index = columns
        return df['count']

    @parse_schema_table
    def get_shape(self, table_name: str, exact: bool = False, schema: Optional[str] = None) -> Tuple[int, int]:
        """Returns the shape of a table

        Args:
            table_name: table name to get shape for
            exact: estimate the count of rows instead of getting the exact value
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            tuple: (n_rows, n_cols)
        """
        if exact:
            stmt = "SELECT COUNT(*) FROM {schema}.{table_name}"
            params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name))
        else:
            self.logger.info('"get_shape" called in estimation mode. Table might need to be "analysed" '
                             'for correct estimation.')
            stmt = "SELECT reltuples::bigint AS estimate FROM pg_class WHERE oid = {fulltable_name}::regclass"
            params = dict(fulltable_name=sql.Literal(f'{schema}.{table_name}'))

        n_rows = self.query(sql.SQL(stmt).format(**params)).iloc[0, 0]
        n_cols = len(self.get_columns(table_name, schema))
        return n_rows, n_cols

    @monitor(mode='time', logger=MONITOR_LOGGER)
    @parse_schema_table
    def get_summary(self, table_name: str, count_null: bool = False, schema: Optional[str] = None) -> pd.DataFrame:
        """Get a summary of the table regarding columns, their type, distinct count and missing values percentage

        Args:
            table_name: table to get summary for
            count_null: Whether to count an existing null value as a distinct value
            schema: Optional. Schema to look for the table in. If None provide will use the instance active_schema

        Returns:
            pd.DataFrame: Table with the different dimensions summarised
        """
        # Compute info
        dtypes = self.get_dtypes(table_name, schema=schema)
        nunique = self.get_nunique(table_name, count_null=count_null, schema=schema)
        nas = self.get_na(table_name, relative=False, schema=schema)
        nas_per = self.get_na(table_name, relative=True, schema=schema)
        df = pd.concat([dtypes, nunique, nas, nas_per], axis=1)
        df.columns = ['type', 'distinct', 'missing_values', 'missing_values_per']
        return df

    def _create_connection(self) -> None:
        self.conn = psycopg2.connect(database=self.name, user=self.user, password=self._password, host=self.host,
                                     port=self.port)
        if self.name is None:
            self.name = self.query('SELECT current_database()').iloc[0, 0]

    def _check_safety(self, stmt, stmt_type):
        if any(x in stmt for x in self._unsafe_symbols):
            raise pd.io.sql.DatabaseError(f'"{stmt_type}" clause passed is unsafe. '
                                          f'Detected use of one of {self._unsafe_symbols}.')

    @staticmethod
    def _check_integrity(values, columns):
        if isinstance(values, pd.DataFrame):
            return

        if columns is None:
            raise KeyError('When "values" is not a DataFrame, "columns" must be passed explicitly.')

        if len(values[0]) != len(columns):
            raise ValueError('When "values" is not a DataFrame, "columns" and "values" must be of matching lengths.')

    @parse_schema_table
    def _commit_table(self, table_name: str, values: Sequence, columns: list, on_conflict: Optional[sql.SQL] = None,
                      update_table_names: bool = True, schema: Optional[str] = None) -> None:
        params = dict(schema=sql.Identifier(schema), table_name=sql.Identifier(table_name),
                      columns=sql.SQL(',').join(map(sql.Identifier, columns)))

        col_stmt = '' if not columns else '({columns})'
        stmt = sql.SQL(f"INSERT INTO {schema}.{table_name} {col_stmt} VALUES %s").format(**params)

        # add on conflict
        if on_conflict is not None:
            stmt = sql.SQL(' ').join((stmt, on_conflict))

        with self.cursor_manager() as cur:
            execute_values(cur, stmt, iter(values))
            self.conn.commit()

            if update_table_names:
                self.refresh()

            return True

    @contextmanager
    def _temporary_primary_key(self, new_key: list, table_name: str) -> list:
        # compare new_key and old_key. If different change key in database
        new_key = sorted(new_key)
        old_key = self.get_primary_key_columns(table_name)
        if new_key != old_key:
            self.set_primary_key(table_name, new_key, on_conflict='drop')

        yield new_key

        # try to reset key to the old one. If fails add the new_key as primary key
        if not old_key:
            self.drop_primary_key(table_name)
        elif new_key != old_key:
            if not self.set_primary_key(table_name, old_key, on_conflict='drop'):
                self.set_primary_key(table_name, new_key, on_conflict='drop')
                self.logger.info(f'Could not reset old primary key ({old_key}). New primary key ({new_key}) kept.')

    @check_options(on_new_columns=('raise', 'add', 'ignore'))
    @parse_schema_table
    def _update_table_schema(self, table_name: str, values: Union[pd.DataFrame, Sequence], columns: Union[dict, list],
                             on_new_columns: str = 'raise', schema: Optional[str] = None) -> pd.DataFrame:
        is_dataframe = isinstance(values, pd.DataFrame)
        types = columns if isinstance(columns, dict) else {}
        orig_types = types.copy()
        columns = list(columns or [])

        # if values is a DataFrame get all definitions from it and ignore passed definitions.
        if is_dataframe:
            if columns:
                self.logger.warning('When passing a DataFrame the "columns" argument is ignored. '
                                    '\n\t\tSpecific types need to be defined in the DataFrame columns. '
                                    '\n\t\tTo filter the DataFrame, do it before sending it to the method call.')
            types = _get_col_types_from_df_schema(pd.io.sql.get_schema(values, '_'))
            columns = values.columns.to_list()
        else:
            self._check_integrity(values, columns)

        # Find which columns are new and which were already defined
        existing_cols = self.get_columns(table_name, schema)
        cols_keep = [col for col in columns if col in existing_cols]  # exist on both sides
        new_cols = list(set(columns).difference(existing_cols))

        # WARNINGS and ERRORS
        # if there are column definitions for columns already defined set a warning
        cols_prev_def = [col for col in orig_types if col in existing_cols]
        if cols_prev_def:
            self.logger.warning(f'Columns {cols_prev_def} were already defined previously. '
                                f'Their types will not be updated.')

        # if there are new columns which don't have definition raise an error
        new_cols_no_def = [col for col in new_cols if col not in types]
        if new_cols_no_def and (on_new_columns == 'add'):
            raise ValueError(f"Columns {new_cols_no_def} doesn't exist in table and have not been defined in "
                             f"the method call.\nCannot define column in database. Pass explicitly the type of "
                             f"the columns or add the column before.")

        # handle new columns
        if new_cols:
            if on_new_columns == 'add':
                new_cols_no_def = [col for col in new_cols if col not in orig_types]
                if new_cols_no_def and is_dataframe:
                    self.logger.warning(f'Column types for new columns {tuple(new_cols_no_def)} will be inferred '
                                        f'from data.\nType inference can be suboptimal.')

                new_cols_types = {c: t for c, t in types.items() if c in new_cols}
                self.add_columns(table_name, new_cols_types, schema=schema)
                cols_keep += new_cols

            elif on_new_columns == 'raise':
                cols_keep += new_cols  # keeping all columns without adding them will raise an error

            elif on_new_columns == 'ignore' and not is_dataframe:
                self.logger.warning('Since values passed are not in DataFrame format it is not possible to ignore '
                                    'new columns effectively. \nIf new columns exist an error will be raised.')
                cols_keep += new_cols

        # Filter columns if they are to ignore
        if is_dataframe:
            cols_keep = values.columns.intersection(cols_keep).to_list()
            values = values[cols_keep]

            try:
                values = values.to_numpy(na_value=None)
            except TypeError:
                values = values.to_numpy()

            values = values.tolist()

        return values, cols_keep


def _get_col_types_from_df_schema(schema: str) -> pd.Series:
    """Converts a table schema to DataFrame format

    Args:
        schema: PostGreSQL schema sql statement (usually generated by pd.io.sql.get_schema method)

    Returns:
        pd.Series: Series with list of columns and their types
    """
    cols = ['column_name', 'data_type']
    var_types = schema.lower().replace('"', '').split('(\n')[1].rstrip('\n)').split(',')
    var_types = list(map(lambda x: x.split(' ')[-2:], var_types))
    return pd.DataFrame.from_records(var_types, columns=cols).set_index('column_name')['data_type'].to_dict()
