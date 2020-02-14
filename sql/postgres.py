from ..utils.log_utils import timeit
import pandas as pd
import pandas.io.sql
import psycopg2
import io


# noinspection SqlNoDataSourceInspection,SqlDialectInspection
class PostgresDatabase(object):
    """Class to manipulate a PostGreSQL DataBase"""

    def __init__(self, name, user='closer', host='localhost', port='5432'):
        """Instance constructor. Establishes the connection parameters to the DB and checks if there are any tables
        already there and current schemas

        :param string name: Database name
        :param string user: Database user
        :param string host: Database host
        :param string port: Database port to connect through
        """

        self.name = name
        self.user = user
        self.host = host
        self.port = port
        self._cur = None
        self._conn = None
        self._is_connection_open = False
        self._initialize_conn()

        self.current_column_types = dict()
        self.tables = None
        self.update_existing_table_names()

    def _change_column_types(self, table_name, table):
        """Updates existing table schema with changes

        :param string table_name: table name to update schema
        :param pandas.DataFrame table: table with new schema
        :return: None
        """

        _flag = 0
        if not self._is_connection_open:
            _flag = 1
            self._db_connect_open()

        cols_schema = ',\n '.join(table.apply(lambda x: f"""ALTER COLUMN {x.name} TYPE {x['type']}""", axis=1))
        sql = f"""ALTER TABLE "{table_name}" {cols_schema}; """
        self.action(sql)

        # Update current table schemas
        self.current_column_types[table_name] = table.copy(deep=True)

        if _flag == 1:
            self._db_connect_close()

    def _compare_column_types(self, table_name, table):
        """Compare column types of a table to be uploaded/upserted with a possibly already existing one

        :param string table_name: table to compare columns
        :param pandas.DataFrame table: new table to be uploaded/upserted
        :return: None
        """

        schema = pd.io.sql.get_schema(table, table_name, con=self._conn).replace('INTEGER', 'bigint')

        if table_name not in self.tables:
            self.action(schema)
            self.tables = self._get_existing_table_names()

        if table_name in self.current_column_types:
            new_schema = eval_table_schema(self.current_column_types[table_name], convert_table_schema_to_df(schema))

            if new_schema is not None:
                self._change_column_types(table_name, new_schema)

    def _db_connect_open(self):
        """Opens the connection between Python and the DB

        :return: None
        """

        self._conn = psycopg2.connect(dbname=self.name, user=self.user, host=self.host, port=self.port)
        self._cur = self._conn.cursor()
        self._set_connection_status(open_=True)

    def _db_connect_close(self):
        """Closes the connection between Python and the Database

        :return: None
        """

        self._conn.close()
        self._cur.close()
        self._set_connection_status(open_=False)

    def _get_current_column_types(self):
        """Query's the Database for the current schemas for each table and stores them as a dictionary of pandas
        DataFrames in the format: {'table_name': DataFrame}, where each DataFrame has a single column 'type' and its
        index is composed of the column names.

        :return dict _cur_col_types: Dictionary with each table schema in DataBase
        """

        _flag = 0
        if not self._is_connection_open:
            _flag = 1
            self._db_connect_open()

        _cur_col_types = dict()
        for table_name in self.tables:
            sql = f"""SELECT column_name, data_type FROM information_schema.columns where 
                        table_name = '{table_name}'; """
            table = self.query(sql)

            table['column_name'] = table['column_name'].apply(lambda x: f""" "{x}" """)
            table.rename({'column_name': 'var', 'data_type': 'type'}, axis=1, inplace=True)
            table.set_index('var', inplace=True)
            _cur_col_types[table_name] = table

        if _flag == 1:
            self._db_connect_close()

        return _cur_col_types

    def _get_existing_table_names(self):
        """Query's the Database for the current existing tables and returns them as a list of strings

        :return list _tables: List with all table names in DataBase
        """

        _flag = 0
        if not self._is_connection_open:
            _flag = 1
            self._db_connect_open()

        self._cur.execute("SELECT relname FROM pg_class WHERE relkind='r' AND relname !~ '^(pg_|sql_)';")
        _tables = [table[0] for table in self._cur.fetchall()]

        if _flag == 1:
            self._db_connect_close()

        return _tables

    def _initialize_conn(self):
        """Tests the DataBase parameters passed in the Instance constructor

        :return: None
        """

        try:
            self._db_connect_open()
        except psycopg2.OperationalError as e:
            if e.args[0] == 'invalid port number':
                raise ValueError('Invalid "port" used.')
            elif e.args[0].__contains__('Unknown host'):
                raise ValueError('Invalid "host" used.')
            elif e.args[0].__contains__('role'):
                raise ValueError('Invalid "user" used.')
            elif e.args[0].__contains__('database'):
                raise ValueError('Invalid "database" used. Database passed does not exist.')
        else:
            self._db_connect_close()

    def _set_connection_status(self, open_=False):
        """Handles the update of the flag attribute '_is_connection_open' to control whether the connection to the
        Database is currently opened or not

        :param bool open_: If connection just opened then pass True, otherwise, if just closed, pass False
        :return: None
        """

        if open_:
            self._is_connection_open = True
        else:
            self._is_connection_open = False

    def _upload_table(self, table_name, table):
        """Uploads a table with the name passed to the DataBase. If a table with same name already exists, it is
        entirely replaced by the new one. Attributes with table_names and table_schemas are updated

        :param string table_name: table name to upload to in DataBase
        :param pandas.DataFrame table: table to upload
        :return: None
        """

        time = timeit(0)
        self._db_connect_open()

        self._cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        self._compare_column_types(table_name, table)

        values = io.StringIO(table.to_csv(header=False, index=False, na_rep=''))
        sql = f"""COPY "{table_name}" FROM STDIN \
                        WITH CSV \
                        DELIMITER AS ',' \
                        NULL AS ''; """

        self._cur.copy_expert(sql, values)
        self._conn.commit()

        self.update_existing_table_names()
        self._db_connect_close()
        timeit(time)

    def action(self, sql):
        """Performs an action in the DataBase

        :param string sql: sql query/action to be performed
        :return: None
        """

        _flag = 0
        time = 0
        if not self._is_connection_open:
            time = timeit(0)
            _flag = 1
            self._db_connect_open()

        self._cur.execute(sql)
        self._conn.commit()

        if _flag == 1:
            self._db_connect_close()
            timeit(time)

    def append_to_table(self, table_name, table):
        """Append table passed to existing table in DataBase. If the table doesn´t exist, create a new one. If there
        is a primary key set, it will return an error on duplicates conflict. Otherwise appends.
        If there are new columns on the table to append, error will be raised: todo check if columns are the same

        :param string table_name: table name to upload to in DataBase
        :param pandas.DataFrame table: table to upload
        :return: None
        """

        # todo add parameter id_column_pkey
        # Should try to set primary key before adding to table to guarantee no duplicates

        if table_name not in self.tables:
            self._upload_table(table_name, table)

        else:
            time = timeit(0)
            self._db_connect_open()

            # todo check if columns are the same. if new columns add and update schema

            # Check if schema is the same. If needed change for upcasted types
            self._compare_column_types(table_name, table)

            values = io.StringIO(table.to_csv(index=False, na_rep=''))
            columns = ', '.join([f'"{col}"' for col in table.columns])

            sql = f"""COPY "{table_name}" ({columns}) \
                    FROM STDIN \
                    WITH CSV \
                    HEADER \
                    DELIMITER AS ',' \
                    NULL AS ''; """

            self._cur.copy_expert(sql, values)
            self._conn.commit()

            self._db_connect_close()
            timeit(time)

    def drop_primary_key(self, table_name):
        """Drops current primary key

        :param string table_name: table name to drop primary key from
        :return: None
        """

        sql = f'ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_pkey;'
        self.action(sql)

    def get_entire_database(self):
        """Loads entire database to python environment

        :return: Dictionary of format {'table_name': table}
        """

        time = timeit(0)
        tables = {table_name: self.get_table(table_name) for table_name in self.tables}

        print('Overall:')
        timeit(time)

        return tables

    def get_table(self, table_name):
        """Loads table from Database to python environment as pandas.DataFrame

        :param string table_name: name of table to load
        :return pandas.DataFrame: table as DataFrame
        """

        time = timeit(0)
        self._db_connect_open()

        table = pd.read_sql(f"SELECT * FROM {table_name};", self._conn)

        self._db_connect_close()
        timeit(time)
        return table

    def query(self, sql):
        """Perform a query to DataBase

        :param string sql: query to perform
        :return pandas.DataFrame: Result of query as pandas DataFrame
        """

        _flag = 0
        time = 0
        if not self._is_connection_open:
            time = timeit(0)
            _flag = 1
            self._db_connect_open()

        table = pd.read_sql(sql, self._conn)

        if _flag == 1:
            self._db_connect_close()
            timeit(time)
        return table

    def set_primary_key(self, table_name, id_column):
        """Adds a primary key to the table

        :param string table_name: table to add the primary key to
        :param string id_column: column to be defined as primary key. Column must exist in table, otherwise it will
        raise an error
        :return: None
        """

        sql = f'ALTER TABLE {table_name} ADD PRIMARY KEY ("{id_column}"); '

        _flag = 0
        if not self._is_connection_open:
            _flag = 1
            self._db_connect_open()

        self.drop_primary_key(table_name)
        self.action(sql)

        if _flag == 1:
            self._db_connect_close()

    def update_existing_table_names(self):
        """Fetches existing table names and schemas from DataBase

        :return: None
        """

        self.tables = self._get_existing_table_names()
        self.current_column_types = self._get_current_column_types()

    def upload_table(self, table_name, table, mode='append_to_table', **kwargs):
        """Uploads table to Database. Depending on the mode chosen it will replace, append or upsert.

        :param string table_name: table name to upload
        :param pandas.DataFrame table: table to upload
        :param string mode: mode how to upload. default: 'append_to_table'. One of ['append_to_table', 'replace',
        'upsert']
        'replace' - if table already exists in DataBase, drops it and uploads the new one. Else, creates the table
        'append_to_table' - if table already exists in DataBase, appends the new values to existing table. Else,
        creates the table
        'upsert' - if table already exists in DataBase, appends new values and on conflict on values in the primary
        key value, replaces the existing values in DataBase with the new ones. Else, creates the table.
        :param kwargs: if mode == 'upsert' then it is necessary to indicate what is the primary key to resolve
        conflicts on. This information is passed as an argument of format 'id_column_pkey = {column_name}'.
        {column_name} must be an existing column of the table to upsert to
        :return: None
        """

        funcs = {'append_to_table': self.append_to_table, 'replace': self._upload_table, 'upsert': self.upsert}
        assert mode in funcs.keys(), \
            KeyError(f'"mode" must be one of the following: {", ".join(str(x) for x in funcs.keys())}. Default is '
                     f'"append_to_table".')

        if mode == 'upsert':
            funcs[mode](table_name, table, **kwargs)
        else:
            funcs[mode](table_name, table)

    def upsert(self, table_name, table, id_column_pkey=None):
        """If table already exists in DataBase, appends new values and on conflict on values in the primary key value,
        replaces the existing values in DataBase with the new ones. Else, creates the table.

        :param string table_name: table name to upload
        :param pandas.DataFrame table: table to upload
        :param string id_column_pkey: column name to be used as primary key
        :return: None
        """

        if table_name not in self.tables:
            self._upload_table(table_name, table)

        else:
            assert id_column_pkey is not None, KeyError('"id_column_pkey" value needed to perform an upsert.')
            assert id_column_pkey in table.columns, \
                KeyError('"id_column_pkey" must be one of the columns of the passed "table".')

            time = timeit(0)
            self._db_connect_open()
            # todo check if columns are the same. if new columns add and update schema

            # Check if schema is the same. If needed change for upcasted types
            self._compare_column_types(table_name, table)

            # todo save existing primary keys; also, id_column_pkey becomes optional. if none passed check if
            #  there is any existing key and use it
            # Set new primary key
            self.set_primary_key(table_name, id_column_pkey)

            # Create sql query and string of values to pass
            sql = create_update_query(table_name, table, id_column_pkey)
            self.action(sql)

            # todo reset primary key to original
            # Remove primary key
            self.drop_primary_key(table_name)
            self._db_connect_close()
            timeit(time)


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
def create_update_query(table_name, table, id_column):
    """This function creates an upsert query which replaces existing data based on primary key conflicts

    :param string table_name: string holding table name to upsert to
    :param pandas DataFrame table: table to upsert to DB
    :param string id_column: primary_key column name
    :return string: query to send to DB
    """

    columns = ', '.join([f'"{col}"' for col in table.columns])
    values = str(list(table.itertuples(index=False, name=None))).strip('[]')
    constraint = f'"{id_column}"'
    excluded = ', '.join(f'EXCLUDED."{col}"' for col in table.columns)

    query = f"""INSERT INTO "{table_name}" ({columns})
                VALUES {values}
                ON CONFLICT ({constraint})
                DO UPDATE SET ({columns}) = ({excluded});"""

    query = ' '.join(query.split())
    return query


def convert_table_schema_to_df(schema):
    """Converts a table schema to DataFrame format

    :param string schema: PostGreSQL schema
    :return pandas.DataFrame: DataFrame with type of each column in schema
    """

    df = pd.DataFrame(schema.split('(\n')[1].rstrip('\n)').split(','), columns=['var_type'])
    df['var'], df['type'] = zip(*df['var_type'].apply(lambda x: x.split(" ")[-2:]))
    df.set_index('var', inplace=True)

    return df['type'].to_frame()


def eval_table_schema(current_schema, new_schema):
    """Compares two PostGreSQL schemas. If there are changes from current_schema to new_schema, updates the types of
    columns of the current schema to be able to contain the types of the new_schema.
    Scale implemented: TEXT > REAL > bigint > INTEGER

    :param pandas.DataFrame current_schema: current PostGreSQL table schema
    :param pandas.DataFrame new_schema: new PostGreSQL table schema
    :return: None or DataFrame with changes in schema to upload
    """

    casting_order = {'TEXT': 1, 'REAL': 2, 'bigint': 3, 'INTEGER': 4}
    reverse_order = {v: k for k, v in casting_order.items()}

    if new_schema.equals(current_schema):
        return None
    else:
        _temp = pd.merge(current_schema, new_schema, how='outer', left_index=True, right_index=True,
                         suffixes=('_cur', '_new'))
        _temp = _temp[_temp['type_cur'] != _temp['type_new']]
        _temp[['type_cur', 'type_new']] = _temp[['type_cur', 'type_new']].replace(casting_order)
        _temp['type_final'] = _temp[['type_cur', 'type_new']].min(axis=1).astype(int)

        if _temp.empty:
            return None
        else:
            _temp['type_final'] = _temp['type_final'].replace(reverse_order)
            return _temp['type_final'].to_frame('type')
