import warnings
from typing import Optional, Sequence

import cx_Oracle
import pandas as pd

from .base import BaseActor
from ..utils.os_utl import check_types, NoneType


class OracleManager(BaseActor):
    """This class provides methods to interact with an Oracle Database

    Arguments:
        :arg user: User identification in the oracle database
        :arg password: User password to the oracle database
        :arg name: Name of the oracle database to connect to. Usually something like '<host>:<port>' or a name
        defined in the tnsnames.ora file
        :arg host: Address to the oracle database (if 'name' is not provided)
        :arg port: Port to connect through (if 'name' is not provided)
        :arg kwargs: Additional keyword arguments to pass to BaseActor class (active_schema)

    """

    def __init__(self, user: str, password: str, name: Optional[str] = None, host: Optional[str] = None,
                 port: Optional[str] = None, **kwargs) -> None:
        name = self._prepare_name(name, host, port)
        self._flavor = 'OracleDB'
        super().__init__(name, user, password, **kwargs)

        self.tables = self.get_table_names()

    def _prepare_name(self, name: str, host: str, port: str) -> str:
        if name is not None:
            return name
        else:
            if (host is not None) and (port is not None):
                return f'{host}:{port}'
            else:
                raise KeyError(f'[{self._flavor}] Either "name" or both "host" and "port" need to be provided to '
                               f'establish a connection.')

    def _open_connection(self):
        self._conn = cx_Oracle.connect(f'{self.user}/{self._password}@{self.name}')
        if self.active_schema is not None:
            self._conn.current_schema = self.active_schema

        super()._open_connection()

    def get_schemas(self):
        return list(self.query('SELECT DISTINCT owner FROM all_tables')['OWNER'])

    def set_active_schema(self, schema: Optional[str] = None) -> None:
        """Sets the active schema in the database. Any query will be done within the active schema without need
        to specifically identify the schema on the query

        :param schema: string name of the schema to set active
        :return:
        """
        if (schema in self.get_schemas()) or (schema is None):
            super().set_active_schema(schema)
        else:
            warnings.warn(f'\n[{self._flavor}] Passed schema "{schema}" does not exist in database "{self.name}" or '
                          f'current user might not have access privileges to it. Schema was not changed.'
                          f'\nCurrent schema: {self.active_schema}', stacklevel=2)

    # todo change for _update_table_names
    def get_table_names(self) -> list:
        """Lists the tables existing in the active schema. If no active schema is set it will list all tables in
        the database to which the user has access

        :return: list of table names
        """
        schemas_and_tables = self.query('SELECT DISTINCT owner, table_name FROM all_tables')
        if self.active_schema is not None:
            return list(schemas_and_tables.loc[schemas_and_tables['OWNER'] == self.active_schema, 'TABLE_NAME'])
        else:
            return list(schemas_and_tables['OWNER'].str.cat(schemas_and_tables['TABLE_NAME'], sep='.'))

    @check_types(table_name=str, limit=(int, NoneType))
    def read_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Extracts a table from the database with table_name. If limit is provided it will only extract a
        specific amount of rows from the top of the database

        :param table_name: name of the table to extract
        :param limit: amount of rows to extract
        :return: pandas DataFrame with the query result
        """
        if limit is None:
            return super().read_table(table_name)

        else:
            query = f"SELECT * FROM {table_name} WHERE ROWNUM <= {limit}"
            return self.query(query)

    def read_table_in_daterange(self, table_name: str, date_column: str, start_date: str, end_date: str,
                                columns: Optional[Sequence] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Extracts columns from table between to dates to be evaluated on a specific column. A limit can be passed,
        If no 'columns' provided it will extract all.

        :param table_name: name of table to query
        :param date_column: name of column to evaluate dates on
        :param start_date: start date for the query
        :param end_date: end data for the query
        :param columns: columns to extract
        :param limit: limit of rows to extract
        :return: DataFrame with the query result
        """
        columns = ', '.join(columns) if columns else '*'
        query = f"SELECT {columns} FROM {table_name} " \
                f"WHERE {date_column} > TO_DATE('{str(start_date)}','YYYY-MM-DD HH24:MI:SS') " \
                f"AND {date_column} < TO_DATE('{str(end_date)}','YYYY-MM-DD HH24:MI:SS')"

        if limit is not None:
            query += f' AND ROWNUM <= {limit}'

        return self.query(query)
