import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from ml_toolkit.feature_encoding import Encoder


if __name__ == '__main__':

    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    X.iloc[:, :] = np.where(X.isnull(), np.nan, X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    enc = Encoder(handle_missing=True, return_as='dataframe', std_categoricals=True, cont_enc='MinMaxScaler',
                  weights={'pclass': 3, 'age': 5}, n_jobs=1, verbose=1)
    exclude = ['name', 'ticket', 'cabin', 'boat', 'home.dest']
    feature_types = {'continuous': [],
                     'categorical': [],
                     'ordinal': ['pclass']}

    encoded = enc.fit_transform(X_train, y=y_train, exclude=exclude, remainder='drop',
                                feature_types=feature_types)
    enc_test = enc.transform(X_test, y_test)
