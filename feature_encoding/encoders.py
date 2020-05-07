from itertools import chain
from functools import partial
from typing import Iterable, Optional, Callable, Union
from contextlib import suppress
from inspect import isfunction, getfullargspec
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy.sparse as sp_sparse
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer

try:
    from .base import EncoderMixin, CategoricalEncoder, __ALL_IMPLEMENTED_ENCODERS__
    from ..utils.os_utl import check_options, check_types
    from ..utils.stats_utl import std_sparse
    from ..utils.generic_utl import duplicated
except (ImportError, ValueError):
    from feature_encoding.base import EncoderMixin, CategoricalEncoder, __ALL_IMPLEMENTED_ENCODERS__
    from utils.os_utl import check_options, check_types
    from utils.stats_utl import std_sparse
    from utils.generic_utl import duplicated
NoneType = type(None)


class Encoder(EncoderMixin):
    """
    todo
    """

    @check_options(return_as=('sparse', 'dataframe', 'array'))
    @check_types(weights=(dict, NoneType), enc_params=(dict, NoneType))
    def __init__(self,
                 cont_enc: str = 'StandardScaler',
                 cat_enc: str = 'OneHotEncoder',
                 multi_cat_enc: str = 'MultiLabelBinarizer',
                 enc_params: Optional[dict] = None,
                 weights: Optional[dict] = None,
                 std_categoricals: bool = False,
                 handle_missing: bool = False,
                 y_transformer: Optional[str, Callable, type] = None,
                 return_as: str = 'sparse',
                 n_jobs: int = 1,
                 verbose: Union[int, bool] = 0) -> None:

        super().__init__(cont_enc, cat_enc, multi_cat_enc, enc_params, std_categoricals, handle_missing)
        self.encoded_feature_names = None
        self.encoder = None
        self.encoder_y = None
        self.input_features = None
        self.input_index = None
        self.feature_types = dict()
        self.fitted = False
        self.weights = weights or {}

        self._exclude = None
        self._feature_types = None
        self._n_jobs = n_jobs
        self._remainder = None
        self._return_as = return_as
        self._verbose = verbose
        self._weights = None

        self.set_encoder_y(y_transformer)

    def set_encoder_y(self, encoder: Optional[str, Callable, type]) -> None:
        """Sets the encoder to be used for y.

        :param encoder: a string to identify the encoder on the implemented ones or a encoder instance/class or a
        function. If a function is passed and it implements and inverse transformation it should have an "inverse"
        keyword argument in the function signature.
        :return:
        """
        if encoder is not None:
            # if implements fit_transform, transform and inverse_transform set this as the transformer
            # else, if it is string search in __all_encoders__ dict for transformer
            # else, if it is a callable check if it has "inverse" in the signature
            # even if it doesn't have an inverse, assign it, but warn the user about this
            if all(hasattr(encoder, attr) for attr in ('fit_transform', 'inverse_transform')):
                self.encoder_y = encoder() if isinstance(encoder, type) else encoder

            elif isinstance(encoder, str):
                try:
                    self.encoder_y = __ALL_IMPLEMENTED_ENCODERS__[encoder]()
                except KeyError:
                    raise NotImplementedError(f'Transformer passed for y ({encoder}) is not implemented. Pass it as a '
                                              f'transformer instance instead.')

            elif isfunction(encoder):
                args = getfullargspec(encoder).args + getfullargspec(encoder).kwonlyargs
                if 'inverse' in args:
                    self.encoder_y = FunctionTransformer(encoder, partial(encoder, inverse=True))
                else:
                    warnings.warn('Callable passed as y transformer does not provide an inverse transformation '
                                  '(boolean keyword argument "inverse" is not present in function signature). '
                                  'Inverse transformations will be identity (thus not transforming back to the '
                                  'original feature space.', stacklevel=2)
                    self.encoder_y = FunctionTransformer(encoder)
                setattr(self.encoder_y, 'custom', True)
            else:
                raise ValueError(f'"y" transformer passed is not applicable. Either pass a transformer, a callable or '
                                 f'one of {__ALL_IMPLEMENTED_ENCODERS__}.')

    def _construct_encoder(self):
        _feature_types = {feat: type_ for type_, feats in self.feature_types.items() for feat in feats}
        _transformers = [(fname, self._encoders[_feature_types[fname]], [fname]) for fname in self.input_features
                         if fname not in list(self._exclude)]
        _sparse = 1 if self._return_as == 'sparse' else 0
        self.encoder = ColumnTransformer(_transformers, self._remainder, sparse_threshold=_sparse, n_jobs=self._n_jobs,
                                         verbose=self._verbose)

    def _retrieve_features(self, arr, fit=False):
        # todo reformat to allow sparse inputs (use of SparseArray in pandas)
        # need to also extract dtypes for next function
        if isinstance(arr, Series):
            arr = arr.to_frame(arr.name)

        if isinstance(arr, DataFrame):
            features = np.asarray(arr.columns, dtype=object)
            assert features.shape[0] == len(set(features)), 'There are duplicated features in the dataset.'
        else:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arr = pd.DataFrame(arr, columns=[str(i) for i in range(arr.shape[1])]).infer_objects()
            features = np.asarray(arr.columns)

        if self._exclude is not None:
            _not_in_features = self._exclude[~np.in1d(self._exclude, features)]
            if (_not_in_features.size > 0) and fit:
                self._exclude = self._exclude[~np.in1d(self._exclude, _not_in_features)]
                warnings.warn(f'Features/indexes {_not_in_features} in exclusion list do not exist in the dataset. '
                              f'Exclusion list was updated.', stacklevel=2)

            features = features[~np.in1d(features, self._exclude)]
            if self._remainder == 'passthrough':
                features = np.append(features, self._exclude)

        if fit:
            self.input_features = features
            self.input_index = arr.index.values

        return arr[features].copy(), features, arr.index.values

    def _validate_nulls(self, X):
        if not self._handle_missing:
            _x = X.values if isinstance(X, DataFrame) else X
            cond = (_x != _x).sum() if sp_sparse.issparse(_x) else pd.isna(_x).sum()
            assert cond == 0, 'Dataset contains null values but "handle_missing" is set as "False". Cannot encode ' \
                              'with null values.'

    def _validate(self, X, types):
        self._validate_nulls(X)

        # validate return
        if (self._remainder == 'passthrough') and (self._return_as == 'sparse'):
            if set(self._exclude).intersection(self.feature_types['categorical']):
                warnings.warn('Cannot return sparse output with non-numeric columns passed in exclude as '
                              '"passthrough". Will return as dense array.', stacklevel=2)
            self._return_as = 'array'

        # validate feature_types
        if types:
            __accepted_types__ = ('continuous', 'categorical', 'ordinal')
            assert all(key in __accepted_types__ for key in types.keys()), f'"feature_types" keys can only be ' \
                                                                           f'{__accepted_types__}'
            assert all(isinstance(value, (list, tuple)) for value in types.values()), f'"feature_types" values must ' \
                                                                                      f'be either a list or a tuple'
        self._feature_types = types or {}

    def _feature_types_sanity_check(self, df, accepted):
        # Remove registered features non existing in df and perform sanity checks
        self._feature_types = {k: [feat for feat in v if feat in self.input_features]
                               for k, v in self._feature_types.items()}
        _registered_features = list(chain.from_iterable(self._feature_types.values())) if self._feature_types else []

        _duplicated = duplicated(_registered_features)
        if _duplicated:
            raise ValueError(f'Features {_duplicated} are defined more than once in "feature_types".')

        _unregistered_features = set(self.input_features).difference(_registered_features)
        if _unregistered_features:
            warnings.warn(f'Features {_unregistered_features} are not defined in "feature_types". '
                          f'Estimator will try to encode these features according to their current dtype.',
                          stacklevel=2)

        _dtypes_not_supported = list(df[_unregistered_features].select_dtypes(exclude=accepted).columns)
        assert len(_dtypes_not_supported) == 0, f'Features {_dtypes_not_supported} are in a non-supported dtype. ' \
                                                f'Convert them to one of {accepted} or define their type ' \
                                                f'in "features_types" argument.'

    def _set_feature_types(self, df):
        __mapped_types__ = {'continuous': 'float', 'categorical': 'str', 'ordinal': 'category'}
        __types_mapping__ = {'str': 'O'}
        __accepted_dtypes__ = ('O', 'category', 'float', 'int', 'int64')
        self._feature_types_sanity_check(df, __accepted_dtypes__)

        # convert features registered if needed
        for dtype, cols in self._feature_types.items():
            new_type = __mapped_types__[dtype]
            _cols = list(df.loc[:, cols].select_dtypes(exclude=[__types_mapping__.get(new_type, new_type)]).columns)
            if len(_cols) > 0:
                df.loc[:, _cols] = df.loc[:, _cols].astype(new_type)

        for col in df.select_dtypes(include='category').columns:
            df[col].cat.reorder_categories(df[col].dtype.categories, ordered=True, inplace=True)

        # Make sure object columns only contain string
        df[df.select_dtypes(include='O').columns] = df.select_dtypes(include='O').astype('str')

        # save the feature types
        self.feature_types = {'continuous': list(df.select_dtypes(include=['int', 'int64', 'float']).columns),
                              'categorical': list(df.select_dtypes(include='O').columns),
                              'ordinal': list(df.select_dtypes(include='category').columns)}

    def _get_encoded_feature_names(self):
        if self.encoder is None:
            raise NotFittedError('Estimator is not yet fitted. Call "fit" or "fit_transform" methods first.')

        def get_names(tf):
            if isinstance(tf, CategoricalEncoder):
                return tf.feature_names
            return ''

        full_names = []
        for name, trans in self.encoder.named_transformers_.items():
            if isinstance(trans, Pipeline):
                names = (get_names(step_trans) for step_trans in trans.named_steps.values())
                names = list(filter(None, chain.from_iterable(names)))
            else:
                if name == 'remainder':
                    names = self._exclude
                    name = ''
                else:
                    names = get_names(trans)

            names = [f'{name}|{str(col)}'.strip('|') for col in names] if len(names) > 0 else [name]
            full_names.extend(names)

        self.encoded_feature_names = np.asarray(full_names)

    def _calculate_weights(self, X):
        size = len(self.encoded_feature_names)
        self._weights = [self.encoded_feature_names, np.ones(size), np.ones(size)]
        orig_names = np.asarray(list(map(lambda x: x.split('|')[0], self._weights[0])))

        if self.__compute_weights__:
            cat_to_calc = [enc_feat for enc_feat, orig_feat in zip(self.encoded_feature_names, orig_names)
                           if ((orig_feat in self.feature_types['categorical']) and (orig_feat not in self._exclude))]

            cont_std = std_sparse(X[:, np.in1d(orig_names, self.feature_types['continuous'])].astype(float),
                                  axis=0).mean()
            for feat in cat_to_calc:
                index = np.where(self.encoded_feature_names == feat)
                self._weights[1][index] = cont_std / std_sparse(X[:, index])

        # construct custom weights array
        for k, v in self.weights.items():
            self._weights[2][np.where(orig_names == k)] = v

    def _apply_weights(self, X):
        weights = self._weights[1] * self._weights[2]
        if sp_sparse.issparse(X):
            X[:, weights != 1] = X[:, weights != 1].multiply(weights[weights != 1])
        else:
            X[:, weights != 1] *= weights[weights != 1]
        return X

    def _construct_output(self, X, feature_names, index):
        if self._return_as == 'sparse':
            if sp_sparse.issparse(X):
                return X
            else:
                with suppress(Exception):
                    return sp_sparse.csr_matrix(X)
                warnings.warn('Was not able to convert matrix to sparse. Will return as dense', stacklevel=2)

        if sp_sparse.issparse(X):
            X = X.toarray()

        if self._return_as == 'dataframe':
            X = pd.DataFrame(X, columns=feature_names, index=index)

        return X

    def _encode_y(self, y, fit=False, index=None):
        if self.encoder_y is None:
            raise NotImplementedError('Trying to transform "y" input, but no transformer for "y" was set.')
            # todo when no transform for y set and y is passed, find type of y and set standard encoder for that type

        _y = y.copy()

        # if Encoder is not custom, prepare data for usual encoders
        feature_names = None
        if not hasattr(self.encoder_y, 'custom'):
            if isinstance(_y, Series):
                _y = _y.to_frame(_y.name)

            if isinstance(_y, DataFrame):
                feature_names = np.asarray(_y.columns, dtype=object)
                _y = _y.values

            else:
                if _y.ndim == 1:
                    _y = _y.reshape(-1, 1)

        # get values
        output = self.encoder_y.fit_transform(_y) if fit else self.encoder_y.transform(_y)

        # get header
        try:
            feature_names = self.encoder_y.feature_names
        except AttributeError:
            n_cols = 1 if output.ndim == 1 else output.shape[1]
            if (feature_names is None) or (len(feature_names) != n_cols):
                feature_names = list(range(n_cols))

        # get index
        index = index if index is not None else range(len(output))
        return self._construct_output(output, feature_names, index)

    @check_options(remainder=('drop', 'passthrough'))
    @check_types(X=(DataFrame, Series, np.ndarray), y=(Series, np.ndarray, NoneType), exclude=(Iterable, NoneType),
                 feature_types=(dict, NoneType))
    def fit_transform(self, X, y=None,
                      feature_types: Optional[Iterable] = None,
                      exclude: Optional[Iterable] = None,
                      remainder: str = 'drop'):
        """todo finish doc and type hinting

        :param X:
        :param y:
        :param feature_types:
        :param exclude:
        :param remainder:
        :return:
        """

        self._exclude = np.asarray(exclude, dtype=object) if exclude is not None else np.asarray([])
        self._remainder = remainder
        self._validate(X, feature_types)

        X, _, index = self._retrieve_features(X, fit=True)
        self._set_feature_types(X)

        self._construct_encoder()
        result = self.encoder.fit_transform(X)
        self._get_encoded_feature_names()

        self._calculate_weights(result)
        result = self._apply_weights(result)

        result = self._construct_output(result, self.encoded_feature_names, index)
        self.fitted = True

        if y is None:
            return result
        else:
            return result, self._encode_y(y, fit=True, index=index)

    def transform(self, X, y=None):
        """todo finish doc and type hinting

        :param X:
        :param y:
        :return:
        """

        if not self.fitted:
            raise NotFittedError('Estimator is not yet fitted. Call "fit" or "fit_transform" methods first.')

        self._validate_nulls(X)

        X, features, index = self._retrieve_features(X, fit=False)
        if not set(self.input_features).issubset(features):
            raise KeyError('Features in array are not the same as in the dataset used to "fit" the estimator. '
                           'Check the features used during "fit" in attribute "input_features".')

        features = features[np.in1d(features, self.input_features)]
        if not all(np.equal(features, self.input_features)):
            warnings.warn('Dataset passed is not sorted in the same order as the dataset used to "fit". '
                          'The dataset will be sorted according to the input_features.', stacklevel=2)

        X = X[self.input_features]
        result = self._apply_weights(self.encoder.transform(X))
        result = self._construct_output(result, self.encoded_feature_names, index)

        if y is None:
            return result
        else:
            return result, self._encode_y(y, index=index)

    def fit(self, X, y=None, feature_types=None, exclude=None, remainder='drop'):
        """todo finish doc and type hinting

        :param X:
        :param y:
        :param feature_types:
        :param exclude:
        :param remainder:
        :return:
        """
        self.fit_transform(X, y, feature_types, exclude, remainder)
        return self


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    data_X = pd.read_csv('feature_encoding/sick_x.csv', index_col=0)
    data_y = pd.read_csv('feature_encoding/sick_y.csv', index_col=0, squeeze=True)
    data_X.iloc[:, :] = np.where(data_X.isnull(), np.nan, data_X)
    data_X.drop(['TBG', 'TBG_measured'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=20)

    enc = Encoder(handle_missing=True, return_as='dataframe', std_categoricals=False, cont_enc='StandardScaler',
                  weights={'age': 5, 'sex': 10}, n_jobs=1, verbose=1)
    exclusion_list = ['goitre']
    feature_types_dict = {'continuous': [],
                          'categorical': [],
                          'ordinal': ['pclass']}

    encoded = enc.fit_transform(X_train, y=y_train, exclude=exclusion_list, remainder='drop',
                                feature_types=feature_types_dict)
    enc_test = enc.transform(X_test, y_test)
