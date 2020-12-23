from support_toolkit.src.db_interaction import PostgreSQLManager


# todo create proper unittest
if __name__ == '__main__':
    import pandas as pd

    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [1, 2, None]})
    df2 = pd.DataFrame({'a': ['1', 2, 3, 4], 'B': ['z', 'b', 'e', 'd'], 'D': [1, 2, None, 3]})

    db = PostgreSQLManager('postgres', 'sia', name='postgres', host='localhost')
    db.drop_table(['a', 'b'])
    db.upload_table('a', df)
    print(db.read_table('a'))
    db.set_primary_key('a', 'a')
    db.upsert('a', df2, ('a', 'b'))

    print(1)
