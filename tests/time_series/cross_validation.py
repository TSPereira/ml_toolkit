import pandas as pd

from ml_toolkit.time_series import GroupTimeSeriesSplit


if __name__ == '__main__':
    a = pd.DataFrame({'value': [1, 1, 17, 1, 2, 2, 22, 20, 4, 5, 4, 4, 8, 1, 2, 4]})
    b = GroupTimeSeriesSplit(dependent_feature='value', n_splits=4, min_train_size=0.5, test_size=0.1, period=0.2)
    print(a)
    print(list(b.split(a)))
