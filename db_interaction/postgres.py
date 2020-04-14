import warnings
from contextlib import contextmanager

import pandas.io.sql
import psycopg2
import io

from .base import BaseActor
from ..utils.os_utl import check_types, NoneType


# noinspection SqlResolve,SqlNoDataSourceInspection,SqlDialectInspection
class PostGreSQL(BaseActor):
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

        super().__init__(name, user, password, **kwargs)
        self.tables = self.get_table_names()

    @contextmanager
    def connection_manager(self):
        """Creates a connection manager to handle the connection to the oracle database.
        This will close the connection with need to call 'connection.close()'

        :return: None
        """
        if not self._is_connection_open:
            self._conn = psycopg2.connect(dbname=self.name, user=self.user, password=self.password,
                                          host=self.host, port=self.port)
            with super().connection_manager():
                yield

        else:
            yield

    def set_active_schema(self, schema):
        """Sets the active schema in the database. Any query will be done within the active schema without need
        to specifically identify the schema on the query

        :param schema: string name of the schema to set active
        :return:
        """
        if schema in self.get_schemas():
            super().set_active_schema(schema)
            self.action(f'ALTER USER {self.user} SET search_path TO "{self.active_schema}"')
            self.tables = self.get_table_names()
        else:
            warnings.warn(f'\nPassed schema "{schema}" does not exist in database "{self.name}" or current user '
                          f'might not have access privileges to it. Schema was not changed.'
                          f'\nCurrent schema: {self.active_schema}', stacklevel=2)

    def get_table_names(self):
        """Lists the tables existing in the active schema. If no active schema is set it will list all tables in
        the database to which the user has access

        :return: list of table names
        """
        sql = "SELECT DISTINCT table_schema, table_name FROM information_schema.tables " \
              "WHERE table_schema !~ 'pg_catalog|information_schema'"
        schemas_and_tables = self.query(sql)
        if self.active_schema is not None:
            return list(schemas_and_tables.loc[schemas_and_tables['table_schema'] == self.active_schema, 'table_name'])
        else:
            return list(schemas_and_tables['table_schema'].str.cat(schemas_and_tables['table_name'], sep='.'))

    def get_schemas(self):
        """Finds available schemas in the database"""
        sql = "SELECT DISTINCT table_schema FROM information_schema.tables " \
              "WHERE table_schema !~ 'pg_catalog|information_schema'"
        return list(self.query(sql)['table_schema'])

    @check_types(table_name=str, limit=(int, NoneType))
    def get_table(self, table_name, limit=None, alternative_schema=None):
        """Extracts a table from the database with table_name. If limit is provided it will only extract a
        specific amount of rows from the top of the database

        :param table_name: name of the table to extract
        :param limit: amount of rows to extract
        :param alternative_schema: If the table to access is not in the active schema
        :return: pandas DataFrame with the query result
        """
        table_name = f'"{table_name}"' if alternative_schema is None else f'"{alternative_schema}"."{table_name}"'
        if limit is None:
            return super().get_table(table_name)

        else:
            query = f'SELECT * FROM {table_name} LIMIT {limit}'
            return self.query(query)

    def _upload_table(self, table_name, table):
        """Uploads a table with the name passed to the DataBase. If a table with same name already exists, it is
        entirely replaced by the new one. Attributes with table_names and table_schemas are updated

        :param string table_name: table name to upload to in DataBase
        :param pandas.DataFrame table: table to upload
        :return: None
        """

        with self.connection_manager():
            self._cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            # self._compare_column_types(table_name, table)

            values = io.StringIO(table.to_csv(header=False, index=False, na_rep=''))
            sql = f"""COPY "{table_name}" FROM STDIN \
                            WITH CSV \
                            DELIMITER AS ',' \
                            NULL AS ''; """

            self._cur.copy_expert(sql, values)
            self._conn.commit()

            # self.update_existing_table_names()


# Todo complete methods for uploading/upserting

#     def _change_column_types(self, table_name, table):
#         """Updates existing table schema with changes
#
#         :param string table_name: table name to update schema
#         :param pandas.DataFrame table: table with new schema
#         :return: None
#         """
#
#         _flag = 0
#         if not self._is_connection_open:
#             _flag = 1
#             self._db_connect_open()
#
#         cols_schema = ',\n '.join(table.apply(lambda x: f"""ALTER COLUMN {x.name} TYPE {x['type']}""", axis=1))
#         sql = f"""ALTER TABLE "{table_name}" {cols_schema}; """
#         self.action(sql)
#
#         # Update current table schemas
#         self.current_column_types[table_name] = table.copy(deep=True)
#
#         if _flag == 1:
#             self._db_connect_close()
#
#     def _compare_column_types(self, table_name, table):
#         """Compare column types of a table to be uploaded/upserted with a possibly already existing one
#
#         :param string table_name: table to compare columns
#         :param pandas.DataFrame table: new table to be uploaded/upserted
#         :return: None
#         """
#
#         schema = pd.io.sql.get_schema(table, table_name, con=self._conn).replace('INTEGER', 'bigint')
#
#         if table_name not in self.tables:
#             self.action(schema)
#             self.tables = self._get_existing_table_names()
#
#         if table_name in self.current_column_types:
#             new_schema = eval_table_schema(self.current_column_types[table_name], convert_table_schema_to_df(schema))
#
#             if new_schema is not None:
#                 self._change_column_types(table_name, new_schema)
#
#     def _get_current_column_types(self):
#         """Query's the Database for the current schemas for each table and stores them as a dictionary of pandas
#         DataFrames in the format: {'table_name': DataFrame}, where each DataFrame has a single column 'type' and its
#         index is composed of the column names.
#
#         :return dict _cur_col_types: Dictionary with each table schema in DataBase
#         """
#
#         _flag = 0
#         if not self._is_connection_open:
#             _flag = 1
#             self._db_connect_open()
#
#         _cur_col_types = dict()
#         for table_name in self.tables:
#             sql = f"""SELECT column_name, data_type FROM information_schema.columns where
#                         table_name = '{table_name}'; """
#             table = self.query(sql)
#
#             table['column_name'] = table['column_name'].apply(lambda x: f""" "{x}" """)
#             table.rename({'column_name': 'var', 'data_type': 'type'}, axis=1, inplace=True)
#             table.set_index('var', inplace=True)
#             _cur_col_types[table_name] = table
#
#         if _flag == 1:
#             self._db_connect_close()
#
#         return _cur_col_types
#
#     def _get_existing_table_names(self):
#         """Query's the Database for the current existing tables and returns them as a list of strings
#
#         :return list _tables: List with all table names in DataBase
#         """
#
#         _flag = 0
#         if not self._is_connection_open:
#             _flag = 1
#             self._db_connect_open()
#
#         self._cur.execute("SELECT relname FROM pg_class WHERE relkind='r' AND relname !~ '^(pg_|sql_)';")
#         _tables = [table[0] for table in self._cur.fetchall()]
#
#         if _flag == 1:
#             self._db_connect_close()
#
#         return _tables
#
#     def _upload_table(self, table_name, table):
#         """Uploads a table with the name passed to the DataBase. If a table with same name already exists, it is
#         entirely replaced by the new one. Attributes with table_names and table_schemas are updated
#
#         :param string table_name: table name to upload to in DataBase
#         :param pandas.DataFrame table: table to upload
#         :return: None
#         """
#
#         self._db_connect_open()
#
#         self._cur.execute(f"DROP TABLE IF EXISTS {table_name};")
#         self._compare_column_types(table_name, table)
#
#         values = io.StringIO(table.to_csv(header=False, index=False, na_rep=''))
#         sql = f"""COPY "{table_name}" FROM STDIN \
#                         WITH CSV \
#                         DELIMITER AS ',' \
#                         NULL AS ''; """
#
#         self._cur.copy_expert(sql, values)
#         self._conn.commit()
#
#         self.update_existing_table_names()
#         self._db_connect_close()
#

#
#     def append_to_table(self, table_name, table):
#         """Append table passed to existing table in DataBase. If the table doesn´t exist, create a new one. If there
#         is a primary key set, it will return an error on duplicates conflict. Otherwise appends.
#         If there are new columns on the table to append, error will be raised: todo check if columns are the same
#
#         :param string table_name: table name to upload to in DataBase
#         :param pandas.DataFrame table: table to upload
#         :return: None
#         """
#
#         # todo add parameter id_column_pkey
#         # Should try to set primary key before adding to table to guarantee no duplicates
#
#         if table_name not in self.tables:
#             self._upload_table(table_name, table)
#
#         else:
#             self._db_connect_open()
#
#             # todo check if columns are the same. if new columns add and update schema
#
#             # Check if schema is the same. If needed change for upcasted types
#             self._compare_column_types(table_name, table)
#
#             values = io.StringIO(table.to_csv(index=False, na_rep=''))
#             columns = ', '.join([f'"{col}"' for col in table.columns])
#
#             sql = f"""COPY "{table_name}" ({columns}) \
#                     FROM STDIN \
#                     WITH CSV \
#                     HEADER \
#                     DELIMITER AS ',' \
#                     NULL AS ''; """
#
#             self._cur.copy_expert(sql, values)
#             self._conn.commit()
#
#             self._db_connect_close()
#
#     def drop_primary_key(self, table_name):
#         """Drops current primary key
#
#         :param string table_name: table name to drop primary key from
#         :return: None
#         """
#
#         sql = f'ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_pkey;'
#         self.action(sql)
#
#     def get_entire_database(self):
#         """Loads entire database to python environment
#
#         :return: Dictionary of format {'table_name': table}
#         """
#
#         tables = {table_name: self.get_table(table_name) for table_name in self.tables}
#         return tables
#
#     def set_primary_key(self, table_name, id_column):
#         """Adds a primary key to the table
#
#         :param string table_name: table to add the primary key to
#         :param string id_column: column to be defined as primary key. Column must exist in table, otherwise it will
#         raise an error
#         :return: None
#         """
#
#         sql = f'ALTER TABLE {table_name} ADD PRIMARY KEY ("{id_column}"); '
#
#         _flag = 0
#         if not self._is_connection_open:
#             _flag = 1
#             self._db_connect_open()
#
#         self.drop_primary_key(table_name)
#         self.action(sql)
#
#         if _flag == 1:
#             self._db_connect_close()
#
#     def update_existing_table_names(self):
#         """Fetches existing table names and schemas from DataBase
#
#         :return: None
#         """
#
#         self.tables = self._get_existing_table_names()
#         self.current_column_types = self._get_current_column_types()
#
#     def upload_table(self, table_name, table, mode='append_to_table', **kwargs):
#         """Uploads table to Database. Depending on the mode chosen it will replace, append or upsert.
#
#         :param string table_name: table name to upload
#         :param pandas.DataFrame table: table to upload
#         :param string mode: mode how to upload. default: 'append_to_table'. One of ['append_to_table', 'replace',
#         'upsert']
#         'replace' - if table already exists in DataBase, drops it and uploads the new one. Else, creates the table
#         'append_to_table' - if table already exists in DataBase, appends the new values to existing table. Else,
#         creates the table
#         'upsert' - if table already exists in DataBase, appends new values and on conflict on values in the primary
#         key value, replaces the existing values in DataBase with the new ones. Else, creates the table.
#         :param kwargs: if mode == 'upsert' then it is necessary to indicate what is the primary key to resolve
#         conflicts on. This information is passed as an argument of format 'id_column_pkey = {column_name}'.
#         {column_name} must be an existing column of the table to upsert to
#         :return: None
#         """
#
#         funcs = {'append_to_table': self.append_to_table, 'replace': self._upload_table, 'upsert': self.upsert}
#         assert mode in funcs.keys(), \
#             KeyError(f'"mode" must be one of the following: {", ".join(str(x) for x in funcs.keys())}. Default is '
#                      f'"append_to_table".')
#
#         if mode == 'upsert':
#             funcs[mode](table_name, table, **kwargs)
#         else:
#             funcs[mode](table_name, table)
#
#     def upsert(self, table_name, table, id_column_pkey=None):
#         """If table already exists in DataBase, appends new values and on conflict on values in the primary key value,
#         replaces the existing values in DataBase with the new ones. Else, creates the table.
#
#         :param string table_name: table name to upload
#         :param pandas.DataFrame table: table to upload
#         :param string id_column_pkey: column name to be used as primary key
#         :return: None
#         """
#
#         if table_name not in self.tables:
#             self._upload_table(table_name, table)
#
#         else:
#             assert id_column_pkey is not None, KeyError('"id_column_pkey" value needed to perform an upsert.')
#             assert id_column_pkey in table.columns, \
#                 KeyError('"id_column_pkey" must be one of the columns of the passed "table".')
#
#             self._db_connect_open()
#             # todo check if columns are the same. if new columns add and update schema
#
#             # Check if schema is the same. If needed change for upcasted types
#             self._compare_column_types(table_name, table)
#
#             # todo save existing primary keys; also, id_column_pkey becomes optional. if none passed check if
#             #  there is any existing key and use it
#             # Set new primary key
#             self.set_primary_key(table_name, id_column_pkey)
#
#             # Create sql query and string of values to pass
#             sql = create_update_query(table_name, table, id_column_pkey)
#             self.action(sql)
#
#             # todo reset primary key to original
#             # Remove primary key
#             self.drop_primary_key(table_name)
#             self._db_connect_close()
#
#
# # noinspection SqlDialectInspection,SqlNoDataSourceInspection
# def create_update_query(table_name, table, id_column):
#     """This function creates an upsert query which replaces existing data based on primary key conflicts
#
#     :param string table_name: string holding table name to upsert to
#     :param pandas DataFrame table: table to upsert to DB
#     :param string id_column: primary_key column name
#     :return string: query to send to DB
#     """
#
#     columns = ', '.join([f'"{col}"' for col in table.columns])
#     values = str(list(table.itertuples(index=False, name=None))).strip('[]')
#     constraint = f'"{id_column}"'
#     excluded = ', '.join(f'EXCLUDED."{col}"' for col in table.columns)
#
#     query = f"""INSERT INTO "{table_name}" ({columns})
#                 VALUES {values}
#                 ON CONFLICT ({constraint})
#                 DO UPDATE SET ({columns}) = ({excluded});"""
#
#     query = ' '.join(query.split())
#     return query
#
#
# def convert_table_schema_to_df(schema):
#     """Converts a table schema to DataFrame format
#
#     :param string schema: PostGreSQL schema
#     :return pandas.DataFrame: DataFrame with type of each column in schema
#     """
#
#     df = pd.DataFrame(schema.split('(\n')[1].rstrip('\n)').split(','), columns=['var_type'])
#     df['var'], df['type'] = zip(*df['var_type'].apply(lambda x: x.split(" ")[-2:]))
#     df.set_index('var', inplace=True)
#
#     return df['type'].to_frame()
#
#
# def eval_table_schema(current_schema, new_schema):
#     """Compares two PostGreSQL schemas. If there are changes from current_schema to new_schema, updates the types of
#     columns of the current schema to be able to contain the types of the new_schema.
#     Scale implemented: TEXT > REAL > bigint > INTEGER
#
#     :param pandas.DataFrame current_schema: current PostGreSQL table schema
#     :param pandas.DataFrame new_schema: new PostGreSQL table schema
#     :return: None or DataFrame with changes in schema to upload
#     """
#
#     casting_order = {'TEXT': 1, 'REAL': 2, 'bigint': 3, 'INTEGER': 4}
#     reverse_order = {v: k for k, v in casting_order.items()}
#
#     if new_schema.equals(current_schema):
#         return None
#     else:
#         _temp = pd.merge(current_schema, new_schema, how='outer', left_index=True, right_index=True,
#                          suffixes=('_cur', '_new'))
#         _temp = _temp[_temp['type_cur'] != _temp['type_new']]
#         _temp[['type_cur', 'type_new']] = _temp[['type_cur', 'type_new']].replace(casting_order)
#         _temp['type_final'] = _temp[['type_cur', 'type_new']].min(axis=1).astype(int)
#
#         if _temp.empty:
#             return None
#         else:
#             _temp['type_final'] = _temp['type_final'].replace(reverse_order)
#             return _temp['type_final'].to_frame('type')
