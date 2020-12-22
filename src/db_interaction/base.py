import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional

import pandas as pd


# noinspection SqlNoDataSourceInspection,SqlDialectInspection
class BaseActor(ABC):
    """Class to manipulate a DataBase"""
    _flavor = 'NoFlavorSpecifiedDB'

    def __init__(self, name: str, user: str, password: str, schema: Optional[str] = None, **kwargs) -> None:
        """Instance constructor. Establishes the connection parameters to the DB and checks if there are any tables
        already there and current schemas

        :param string name: Database name
        :param string user: Database user
        """

        self._cur = None
        self._conn = None
        self.active_schema = None
        self._is_connection_open = False

        self.name = name
        self.user = user
        self._password = password
        self._connection_test()

        if schema is not None:
            self.set_active_schema(schema)

        self.current_column_types = dict()
        self.tables = None

    def _connection_test(self):
        try:
            with self.connection_manager():
                logging.info(f'[{self._flavor}] Connection Test Successful')
        except Exception as e:
            logging.error(f'[{self._flavor}] Cannot open connection to database with parameters passed.')
            raise e

    @abstractmethod
    @contextmanager
    def connection_manager(self) -> None:
        """
        Must be overriden. To retain the current commands after overriding check the example below:

        Example:
        >>> @contextmanager
        >>> def _connection_manager():
        >>>     # Your new instructions, such as opening the connection to the current db flavor
        >>>     if not self._is_connection_open:
        >>>         self._conn = db_flavor.connect(**kwargs)
        >>>
        >>>         # Call the base class _connection_manager to manage the cursor and connection
        >>>         with super().connection_manager():
        >>>             yield
        >>>
        >>>     # Any additional commands you might want to run after the connection is closed
        :return:
        """

        # On opening
        self._cur = self._conn.cursor()
        self._set_connection_status(open_=True)
        yield

        # on close
        self._cur.close()
        self._conn.close()
        self._set_connection_status(open_=False)

    def _set_connection_status(self, open_: bool = False) -> None:
        """Handles the update of the flag attribute '_is_connection_open' to control whether the connection to the
        Database is currently opened or not

        :param bool open_: If connection just opened then pass True, otherwise, if just closed, pass False
        :return: None
        """
        self._is_connection_open = True if open_ else False

    def set_active_schema(self, schema: str) -> None:
        self.active_schema = schema
        logging.info(f'[{self._flavor}] Active schema changed to "{self.active_schema}".')

    def action(self, sql: str) -> None:
        """Performs an action in the DataBase

        :param string sql: sql query/action to be performed
        :return: None
        """

        with self.connection_manager():
            self._cur.execute(sql)
            self._conn.commit()

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Loads table from Database to python environment as pandas.DataFrame

        :param string table_name: name of table to load
        :return pandas.DataFrame: table as DataFrame
        """

        query = f"SELECT * FROM {table_name}"
        return self.query(query)

    def query(self, sql: str) -> pd.DataFrame:
        """Perform a query to DataBase

        :param string sql: query to perform
        :return pandas.DataFrame: Result of query as pandas DataFrame
        """

        with self.connection_manager():
            table = pd.read_sql(sql, self._conn)
        return table
