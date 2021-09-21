# !/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Classes & functions related to mongoDB databases
"""

# =========================================================================================================
# ================================ 0. MODULE
from typing import Union, Optional, Sequence

import pandas as pd
import pymongo
from pymongo import MongoClient

from ..utils.log_utl import monitor, create_logger
from ..utils.os_utl import check_types, NoneType

from ._core import BaseActor

FLAVOR = 'MongoDB'
LOGGER = create_logger(__name__, on_conflict='ignore', fmt=f'[%(asctime)s | {FLAVOR} | %(levelname)s] %(message)s',
                       logger_level=20)
MONITOR_LOGGER = LOGGER.getChild('monitor')
MONITOR_LOGGER.setLevel(30)

# =========================================================================================================
# ================================ 1. CLASSES


class MongoManager(BaseActor):
    """
    Class helping to simplify collection management in MongoDB

    Parameters
    ----------
    host: str, default='localhost'
        host address to connect to

    user, password: Optional[str], default=None
        Credentials to use to connect

    port: Optional[int], default=27017
        Port to use to connect

    name: Optional[str], default='admin'
        Database name to connect to

    kwargs:
        logger: Optional[Logger], default=<module level Logger>
            Logger to use to log messages

    Attributes
    ----------
    conn
        MongoClient instance from pymongo

    database
        Database object from pymongo
    """

    _flavor = FLAVOR

    def __init__(self, *, host: str = 'localhost', user: Optional[str] = None, password: Optional[str] = None,
                 port: Union[int] = 27017, name: Optional[str] = 'admin', **kwargs) -> None:
        self.database = None
        self.host, self.port = host, port
        self._client_additional_kwargs = kwargs
        super().__init__(name, user, password, logger=LOGGER, **kwargs)

    def _create_connection(self) -> None:
        self.conn = MongoClient(host=self.host, port=self.port, username=self.user, password=self._password,
                                authSource=self.name, **self._client_additional_kwargs)
        self.database = self.conn[self.name]

    def change_database(self, name: str):
        """Change the active database

        Args:
            name: string with the name of the database to use

        Returns:
            None
        """
        self.name = name
        self.database = self.conn[name]

    def list_collections(self) -> list:
        """
        List all the collections stored in the database

        Returns
        -------
        collection_names : str
            Name of the collections
        """
        return self.database.list_collection_names()

    def check_collection_existence(self, collection_name: str) -> bool:
        """
        Check the existence of the table

        Parameters
        ----------
        collection_name : str
            Name of the collection

        Returns
        -------
        existence : bool
            If true, the collection does exist in the database
        """
        return self.database[collection_name].count_documents({}) != 0

    def get_collection(self, collection_name: str) -> pymongo.collection.Collection:
        """
        Parameters
        ----------
        collection_name : str
            Name of the collection

        Returns
        -------
        collection
            Collection object from pymongo
        """
        return self.database[collection_name]

    def drop_collection(self, collection_name: str) -> None:
        """
        Delete a collection from database

        Parameters
        ----------
        collection_name : str
            Name of the collection
        """

        collections = self.list_collections()
        if collection_name in collections:
            self.get_collection(collection_name).drop()
            self.logger.info(f"Collection {collection_name} deleted.")
        else:
            self.logger.info(f"Collection {collection_name} doesn't exist so it can't be dropped.")

    def count_documents(self, collection_name: str, filter_dict: dict = None) -> int:
        """
        Count the number of documents in a collection according to the filter

        Parameters
        ----------
        collection_name : str
            Name of the collection

        filter_dict : dict
            Dictionary corresponding to mongodb queries (e.g. {threshold: {'$gt': 0.4}})

        Returns
        -------
        n : int
            Number of documents
        """
        return self.get_collection(collection_name).count_documents(filter_dict or {})

    @monitor(logger=MONITOR_LOGGER)
    @check_types(data=(dict, Sequence, pd.DataFrame))
    def insert_documents(self, data: Union[dict, Sequence[dict], pd.DataFrame], collection_name: str,
                         return_ids: bool = True, replace: bool = False, save_existing: bool = False) -> Optional[list]:
        """
        Insert one or several document(s) in mongoDB

        Parameters
        ----------
        data : dataframe
            Data to write in MongoDB

        collection_name : str
            Name of the corresponding collection in MongoDB

        return_ids: bool, default=True
            If True, returns the ids of the documents inserted

        replace : bool, default=False
            If True, the existing data in collection will be deleted

        save_existing : bool, default=False
            If True, the existing data will be saved as {collection_name}_old

        Returns
        -------
        list
            list of ids of inserted documents
        """

        # Check if collection already exists
        already_exist = self.check_collection_existence(collection_name)

        # Rename collection as collection_old for rollback in case of error
        if already_exist & replace:
            dt_old = self.database[collection_name + "_old"]
            dt_old.drop()
            dt_old = self.database[collection_name]
            dt_old.rename(collection_name + "_old")

        # Select collection
        dt = self.database[collection_name]

        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")

        try:
            if isinstance(data, dict):
                ids = [dt.insert_one(data).inserted_id]
            elif isinstance(data, Sequence):
                ids = dt.insert_many(data).inserted_ids

        # Rollback if error
        except Exception as e:
            if already_exist & replace:
                dt.drop()
                dt_old = self.database[collection_name + "_old"]
                dt_old.rename(collection_name)
                self.logger.info('Error rollback succeeded')
            raise e

        # Delete old
        if not save_existing:
            dt_old = self.database[collection_name + "_old"]
            dt_old.drop()

        self.logger.info(f'Collection {collection_name} uploaded.')
        if return_ids:
            return ids

    @monitor(logger=MONITOR_LOGGER)
    @check_types(columns=(list, NoneType))
    def get_documents(self, collection_name: str, dataframe: bool = False, columns: Optional[list] = None,
                      filter_dict: dict = None, keep_id: bool = False) -> Union[list, pd.DataFrame]:
        """
        Collect a collection (optionally as a dataframe)

        Parameters
        ----------
        collection_name:  str

        dataframe: bool

        columns : list[str]
            List of names of columns to keep

        filter_dict : dict
            Dictionary corresponding to mongodb queries (e.g. {threshold: {'$gt': 0.4}})

        keep_id : bool, default=False
            If True, keep the id column from mongodb

        Returns
        -------
        data
            Either a list of dictionaries or a dataframe
        """

        # if the collection doesn't exist
        if not self.check_collection_existence(collection_name):
            raise ValueError("Empty or non-existing collection")

        columns_ = {}
        columns = columns or []
        filter_dict = filter_dict or {}

        if len(columns) != 0:
            for col in columns:
                columns_[col] = 1

        if not keep_id:
            columns_.update({"_id": 0})

        args = (filter_dict,) if len(columns_) == 0 else (filter_dict, columns_)
        data = list(self.get_collection(collection_name).find(*args))

        if dataframe:
            data = pd.DataFrame(data)

        self.logger.info(f'Collection {collection_name} downloaded.')
        return data
