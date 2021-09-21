import logging
from inspect import getfullargspec
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Tuple, Any, Union

import pandas as pd
from decorator import decorator

from ..utils.log_utl import create_logger, monitor
from ..utils.os_utl import check_options


MONITOR_LOGGER = create_logger(__name__, logger_level=logging.WARNING)


class BaseActor(ABC):
    """Class to manipulate a DataBase

    Args:
        name: name of the database
        user: user to authenticate on database
        password: password of user
        **kwargs: additional keyword arguments passed that are not used

    """
    _flavor = 'NoFlavorSpecifiedDB'
    _exception_handling = 'ignore'

    def __init__(self, name: str, user: str, password: str, logger=None, **kwargs) -> None:
        self.conn = None
        self.debug = False
        self.logger = logger or create_logger(f'sia_db_interaction.databases.{self._flavor}', on_conflict='ignore',
                                              fmt=f'[%(asctime)s | {self._flavor} | %(levelname)s] %(message)s',
                                              logger_level=20)

        self.name = name
        self.user, self._password = user, password
        self.connect()

    def __del__(self):
        self.close_connection()

    @abstractmethod
    def _create_connection(self) -> None:
        """Method to be overriden by Child class. No need to include logs in this method (See 'connect' method)

        Returns:

        """
        ...

    def connect(self) -> None:
        """Connect to the database

        Returns:
            None
        """
        try:
            self._create_connection()
            self.logger.info(f'Connection opened to {self.name}')
        except Exception as e:
            self.logger.error('Cannot open connection to database with parameters passed.')
            self.logger.error(str(e).strip('\n'))
            raise e

    def close_connection(self) -> None:
        """If a connection is open, close it

        Returns:
            None
        """
        if self.conn is not None and not self.conn.closed:
            self.conn.close()
            self.logger.info('Connection closed.')

    def toggle_debug(self):
        self.debug = not self.debug

    @check_options(action=('ignore', 'raise'))
    def set_exception_handling(self, action='ignore'):
        self._exception_handling = action


class BaseActorSQL(BaseActor, ABC):
    """Class to manipulate a DataBase

    Args:
        name: name of the database
        user: user to authenticate on database
        password: password of user
        schema: Schema to set as active
        **kwargs: additional keyword arguments passed that are not used

    """
    _cursor_manager_exceptions = (Exception, KeyboardInterrupt)

    def __init__(self, name: str, user: str, password: str, schema: Optional[str] = None, logger=None,
                 **kwargs) -> None:
        super().__init__(name, user, password, logger, **kwargs)
        self._tables = None
        self.active_schema = None
        self.set_active_schema(schema)

    def set_active_schema(self, schema: Optional[str] = None) -> None:
        """Set the active schema on the class to avoid the necessity of explicitly passing the schema for every
        interaction

        Args:
            schema: If None, set to "public"

        Returns:
            None
        """
        self.active_schema = schema or 'public'
        self.logger.info(f'Active schema changed to {self.active_schema}.')

    @staticmethod
    def get_sql(stmt: str, params: Optional[Tuple[Any]] = None, params_symbol: str = '?') -> str:
        """Generates the sql statement with the params passed. This can be used to simulate the statements before
        sending them.

        Args:
            stmt: Sql statement to apply params to
            params: Params to insert in the sql statement
            params_symbol: Symbol in the sql statement that identifies where each param should go. Default: ?

        Returns:
            String: Sql statement with params applied to
        """
        if params is not None:
            unique = "%PARAMETER%"
            stmt = stmt.replace(params_symbol, unique)
            for v in params:
                stmt = stmt.replace(unique, repr(v), 1)
        return stmt

    @contextmanager
    def cursor_manager(self) -> None:
        """Manager of cursor. Opens a cursor and returns the action to the code that call the method.
        Once it is completed closes the cursor. If the code in between fails, logs the error and rollback the
        connection.

        Returns:
            None
        """

        try:
            with self.conn.cursor() as cur:
                yield cur

        except self._cursor_manager_exceptions as e:
            self.logger.error(str(e).strip('\n'))
            self.conn.rollback()
            if self._exception_handling == 'raise':
                raise e

    @monitor(logger=MONITOR_LOGGER, log_level='debug')
    def execute(self, stmt: str, params: Optional[Union[dict, tuple]] = None, *, return_cursor_metadata: bool = False,
                log: Optional[str] = None) -> Optional[Union[bool, str, object]]:
        """Performs an action in the DataBase

        Args:
            stmt: sql query/action to be performed
            params: Optional params to be applied on the sql statement placeholders
            return_cursor_metadata: Whether to return the cursor object after close
            log: message to log if successful

        Returns:
            bool: If the action is completed without any error return True
            str: If class in 'debug' mode then it will return the sql stmt that would be executed
            object: If user selects 'return_cursor_metadata' then it return the closed cursor object
            None: If there is an exception in the transaction it returns None
        """

        debug_stmt = self.get_sql(stmt, params)
        self.logger.debug(debug_stmt)

        if self.debug:
            return debug_stmt

        else:
            with self.cursor_manager() as cur:
                cur.execute(stmt, params)
                self.conn.commit()

                if log is not None:
                    self.logger.info(log)

                return cur if return_cursor_metadata else True

    @monitor(logger=MONITOR_LOGGER, log_level='debug')
    def query(self, stmt: str, params: Optional[Union[dict, tuple]] = None, **kwargs) -> pd.DataFrame:
        """Perform a query to DataBase and return as a dataframe

        Args:
            stmt: query to perform
            params: Optional params to be applied on the sql statement placeholders
            **kwargs: Additional optional keyword arguments to pass to pandas.read_sql method

        Returns:
            pd.DataFrame: Result of query
        """
        self.logger.debug(self.get_sql(stmt, params))
        return pd.read_sql(stmt, self.conn, params=params, **kwargs)

    def tables(self, schema: Optional[str] = None) -> list:
        """List the existing tables in the connected database

        Args:
            schema: Schema to use to filter down the list of tables displayed

        Returns:
            List: collection of table names in format {schema}.{table_name}

        """
        if schema is None:
            return self._tables

        return [t for t in self._tables if t.startswith(f'{schema}.')]


@decorator
def parse_schema_table(func, *args, **kwargs):
    """Utility decorator to correctly define "schema" and "table_name" passed into a function. This decorator
    deals with both formats:
        - schema=<schema>, table_name=<table_name>
        - schema=None, table_name=<schema>.<table_name>
        - schema=<schema_ignored>, table_name=<schema>.<table_name>

    Args:
        func: function or method being decorated
        *args: args of the function/method. Must contain "table_name" and optionally "schema". If "schema" is absent
            and "table_name" doesn't contain <schema>, then "schema" will be defined as 'public'.
        **kwargs: Any keyword arguments for function/method. Not used.

    Returns:
        Callable: Original function/method with updated args
    """
    dct = dict(zip(getfullargspec(func).args, args))

    if '.' in dct['table_name']:
        dct['schema'], dct['table_name'] = dct.get('table_name').split('.', 1)
    else:
        dct['schema'] = dct.get('schema', None) or getattr(dct.get('self'), 'active_schema', None) or 'public'

    return func(**{**kwargs, **dct})


def join_schema_to_table_name(name: str, schema: str) -> str:
    """Utility function to join a table_name to a schema. If the table name already has a schema returns the original
    name

    Args:
        name: table name to join
        schema: schema name to join to the table name

    Returns:
        String: Reformatted table name
    """
    return '.'.join(filter(None, (schema, name))) if '.' not in name else name
