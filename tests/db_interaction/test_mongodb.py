import logging
import os
import unittest
from copy import deepcopy
from operator import itemgetter

import pandas as pd
from pymongo.collection import Collection, ObjectId
from pymongo.errors import ServerSelectionTimeoutError
from ml_toolkit.db_interaction.api import MongoManager

print({k: v for k, v in os.environ.items() if k.startswith('MONGO')})
CFG = dict(user=os.environ.get('MONGO_USER', None),
           password=os.environ.get('MONGO_PASSWORD', None),
           host=os.environ.get('MONGO_HOST', 'localhost'),
           port=os.environ.get('MONGO_PORT', 27017),
           name=os.environ.get('MONGO_DB', 'test_db'),
           serverSelectionTimeoutMS=5000)


def open_db(cfg=None):
    db = MongoManager(**CFG) if cfg is None else MongoManager(**cfg)
    db.logger.setLevel(logging.ERROR + 1)
    db.logger.setFormattersIsColored(False)
    db.set_exception_handling('raise')
    return db


class DBTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = open_db()

    @classmethod
    def tearDownClass(cls):
        del cls.db


class ConnectionCase(unittest.TestCase):
    def test_connection(self):
        db = open_db()
        self.assertEqual(db.conn.address, (CFG.get('host'), CFG.get('port')))
        self.assertIsInstance(db.conn.server_info(), dict)

    def test_connection_fail(self):
        cfg = CFG.copy()
        cfg['hostname'] = 'lskdjfl'
        cfg['serverSelectionTimeoutMS'] = 2000

        db = open_db(cfg)
        self.assertRaises(ServerSelectionTimeoutError, db.conn[db.name].command, 'ping')


class InsertDocumentsCase(DBTestCase):
    def setUp(self) -> None:
        self.db.drop_collection('test1')

    def tearDown(self) -> None:
        self.db.drop_collection('test1')

    def test_insert_dict(self):
        self.db.insert_documents({'data': [1, 2, 3]}, 'test1')
        self.assertEqual(list(map(itemgetter('data'), self.db.database['test1'].find())), [[1, 2, 3]])

    def test_insert_sequence_of_dicts(self):
        self.db.insert_documents([{'data': [1, 2, 3]}, {'data': ['a', 'b']}], 'test1')
        self.assertEqual(list(map(itemgetter('data'), self.db.database['test1'].find())),
                         [[1, 2, 3], ['a', 'b']])

    def test_insert_dataframe(self):
        df = pd.DataFrame({'data1': [1, 2, 3], 'data2': [4, 5, 6]})
        self.db.insert_documents(df, 'test1')
        self.assertEqual(list(map(itemgetter('data1', 'data2'), self.db.database['test1'].find())),
                         [(1, 4), (2, 5), (3, 6)])

    def test_insert_other_types(self):
        self.assertRaises(TypeError, self.db.insert_documents, [1, 2, 3], 'test1')
        self.assertRaises(TypeError, self.db.insert_documents, [[1, 2, 3], ['a', 'b']], 'test1')

    def test_return_ids(self):
        ids = self.db.insert_documents([{'data': [1, 2, 3]}], 'test1')
        self.assertIsInstance(ids[0], ObjectId)

        ids = self.db.insert_documents([{'data': [1, 2, 3], '_id': 'test_id'}], 'test1')
        self.assertEqual(ids[0], 'test_id')

        self.assertEqual(self.db.insert_documents([{'data': [1, 2, 3]}], 'test1', return_ids=False), None)

    def test_replace(self):
        self.db.insert_documents([{'data': [1, 2, 3]}], 'test1')
        self.db.insert_documents([{'data': [4, 5, 6]}], 'test1', replace=True)
        self.assertEqual(list(map(itemgetter('data'), self.db.database['test1'].find())),
                         [[4, 5, 6]])

        self.db.insert_documents([{'data': [1, 2, 3]}], 'test1')
        self.assertEqual(list(map(itemgetter('data'), self.db.database['test1'].find())),
                         [[4, 5, 6], [1, 2, 3]])

    def test_save_existing(self):
        self.db.insert_documents([{'data': [1, 2, 3]}], 'test1')
        self.db.insert_documents([{'data': [4, 5, 6]}], 'test1', replace=True, save_existing=True)
        self.assertEqual(list(map(itemgetter('data'), self.db.database['test1'].find())),
                         [[4, 5, 6]])
        self.assertEqual(list(map(itemgetter('data'), self.db.database['test1_old'].find())),
                         [[1, 2, 3]])


class GetCollectionCase(DBTestCase):
    def setUp(self) -> None:
        self.db.drop_collection('test1')

    def tearDown(self) -> None:
        self.db.drop_collection('test1')

    def test_no_collection(self):
        self.assertRaises(ValueError, self.db.get_documents, 'another_not_existant_collection')

    def test_get_all(self):
        docs = [{'data': [1, 2, 3]}, {'data': ['a', 'b']}]
        data = deepcopy(docs)
        self.db.insert_documents(data, 'test1')
        self.assertEqual(self.db.get_documents('test1'), docs)

        docs.append({'data': [4, 5, 6, 7, 8]})
        data = deepcopy(docs)
        self.db.insert_documents(data[-1], 'test1')
        self.assertEqual(self.db.get_documents('test1'), docs)

    def test_get_as_dataframe(self):
        docs = [{'data': [1, 2, 3]}, {'data': ['a', 'b']}]
        data = deepcopy(docs)
        self.db.insert_documents(data, 'test1')
        self.assertTrue(self.db.get_documents('test1', dataframe=True).equals(pd.DataFrame(docs)))

        docs = pd.DataFrame([{'data': [1, 2, 3]}, {'data': ['a', 'b']}])
        data = deepcopy(docs)
        self.db.insert_documents(data, 'test1', replace=True)
        self.assertTrue(self.db.get_documents('test1', dataframe=True).equals(docs))

    def test_columns(self):
        docs = [{'a': [1, 2, 3], 'b': [4, 5, 6]},
                {'a': ['a', 'b'], 'b': ['c', 'd', 'e', 'f']}]
        self.db.insert_documents(docs, 'test1')

        columns_set = (['a'], ['b'], ['a', 'b'])
        for columns in columns_set:
            self.assertEqual(self.db.get_documents('test1', columns=columns),
                             [{k: v for k, v in d.items() if k in columns} for d in docs])

    def test_keep_id(self):
        docs = [{'a': [1, 2, 3], 'b': [4, 5, 6]},
                {'a': ['a', 'b'], 'b': ['c', 'd', 'e', 'f']}]
        self.db.insert_documents(docs, 'test1')

        results = self.db.get_documents('test1', keep_id=False)
        [self.assertNotIn('_id', obj) for obj in results]

        results = self.db.get_documents('test1', keep_id=True)
        [self.assertIn('_id', obj) for obj in results]


class MiscCase(DBTestCase):
    def setUp(self) -> None:
        self.db.insert_documents([{'data': [1, 2, 3]}], 'test1')
        self.db.insert_documents([{'data': [4, 5, 6]}, {'data': ['a', 'b']}], 'test2')

    def tearDown(self) -> None:
        self.db.database = self.db.conn[self.db.name]
        self.db.drop_collection('test1')
        self.db.drop_collection('test2')

    def test_list_collections(self):
        self.assertEqual(sorted(self.db.list_collections()), sorted(['test1', 'test2']))
        self.assertEqual(len(self.db.list_collections()), 2)

    def test_check_collection_existence(self):
        self.assertTrue(self.db.check_collection_existence('test1'))
        self.assertFalse(self.db.check_collection_existence('test3'))

    def test_get_collection(self):
        collection = self.db.get_collection('test1')
        self.assertIsInstance(collection, Collection)
        self.assertEqual(collection.name, 'test1')

        # Also works with non_existing collectios
        collection = self.db.get_collection('some_other_collection')
        self.assertIsInstance(collection, Collection)
        self.assertEqual(collection.name, 'some_other_collection')

    def test_drop_collection(self):
        self.assertIn('test1', self.db.list_collections())
        self.db.drop_collection('test1')
        self.assertNotIn('test1', self.db.list_collections())

    def test_count_documents(self):
        self.assertEqual(self.db.count_documents('test1'), 1)
        self.assertEqual(self.db.count_documents('test2'), 2)
        self.assertEqual(self.db.count_documents('some_other_collection'), 0)

    def test_change_database(self):
        original_name = self.db.name
        self.assertEqual(self.db.database.name, self.db.name)
        self.db.change_database('another_test_db')
        self.assertEqual(self.db.database.name, 'another_test_db')
        self.db.change_database(original_name)
