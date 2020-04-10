from itertools import chain
from collections import ChainMap
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy as sp
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, MinMaxScaler, OrdinalEncoder, \
    StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer

from ..utils.os_utl import check_options, check_types
from ..utils.generic_utl import duplicated, std_sparse
NoneType = type(None)


__CONT_ENCODERS__ = {'MinMaxScaler':   MinMaxScaler,
                     'StandardScaler': StandardScaler,
                     'RobustScaler':   RobustScaler}
__ORD_ENCODERS__ = {'OrdinalEncoder': OrdinalEncoder}
__CAT_ENCODERS__ = {'LabelBinarizer': LabelBinarizer,
                    'OneHotEncoder':  OneHotEncoder}
__MULTI_CAT_ENCODERS__ = {'MultiLabelBinarizer': MultiLabelBinarizer,
                          'CountVectorizer':     CountVectorizer,
                          'FeatureHashing':      FeatureHasher}


class CategoricalEncoder:
    """

    """

    __encoders__ = {'categorical': __CAT_ENCODERS__,
                    'multi_categorical': __MULTI_CAT_ENCODERS__}
    # todo define default enc params per encoder as global variable
    __default_enc_params__ = {'categorical': {'handle_unknown': 'ignore'},
                              'multi_categorical': {'lowercase': False, 'analyzer': list,  # CountVectorizer
                                                    'input_type': 'string'}}               # FeatureHashing

    def __init__(self, cat_enc='OneHotEncoder', multi_cat_enc='MultiLabelBinarizer', enc_params=None, sparse=True):
        # filter enc_params to only have 'categorical' and 'multi_categorical'
        # define default params
        self.__chosen__ = {'categorical': cat_enc, 'multi_categorical': multi_cat_enc}

        self.encoder = None
        self._sparse = sparse
        self._enc_params = {}
        self._type = None
        self.set_encoder_params(enc_params)

    @property
    def _chosen(self):
        return self.__chosen__

    def _get_encoder_params(self, enc_type):
        _expected_args = self.__encoders__[enc_type][self._chosen[enc_type]].__init__.__code__.co_varnames
        return {k: v for k, v in self._enc_params[enc_type].items() if k in _expected_args}

    @check_types(params=(NoneType, dict))
    def set_encoder(self, encoder, feature_type, params=None):
        assert feature_type in self.__encoders__, f'"feature_type" must be one of {list(self.__encoders__.keys())}.'
        assert encoder in self.__encoders__[feature_type], f'"encoder" of type {feature_type} must be one of ' \
                                                           f'{self.__encoders__[feature_type].keys()}.'

        if params:
            self.set_encoder_params({feature_type: params})
        self.__chosen__[feature_type] = encoder

    @check_types(enc_params=(dict, NoneType))
    def set_encoder_params(self, enc_params):
        enc_params = enc_params or {}
        for enc_type, params in self.__default_enc_params__.items():
            assert isinstance(params, dict), 'Parameters for each "feature_type" must be passed as a dictionary.'
            self._enc_params[enc_type] = dict(ChainMap(enc_params.get(enc_type) or {},
                                                       self._enc_params.get(enc_type) or {},
                                                       self.__default_enc_params__[enc_type]))

    def _construct_encoder(self, X):
        multi_cat = any(isinstance(val, list) for val in np.asarray(X).reshape(-1, ))
        self._type = 'multi_categorical' if multi_cat else 'categorical'

        # reshape inputs to correct format
        X = self._reshape_inputs(X, multi_cat)

        # get parameters for encoder chosen
        params = self._get_encoder_params(self._type)
        if (self._chosen[self._type] == 'FeatureHashing') and ('n_features' not in params):
            params['n_features'] = len(set(chain.from_iterable(X)))
            warnings.warn(f'FeatureHashing needs to set a defined number of features. None was explicitly passed. To '
                          f'avoid collisions, "n_features" will be set to the number of unique categories in the '
                          f'column ({params["n_features"]})')

        self.encoder = self.__encoders__[self._type][self._chosen[self._type]](**params)
        return X

    @staticmethod
    def _reshape_inputs(X, multi_cat):
        return list(map(lambda x: [str(x)] if not isinstance(x, list) else [str(y) for y in x], X)) if multi_cat else \
            np.asarray(X).reshape(-1, 1)

    def _construct_output(self, X):
        if self._sparse and not sp.sparse.issparse(X):
            X = sp.sparse.csr_matrix(X)
        elif not self._sparse and sp.sparse.issparse(X):
            X = X.toarray()
        return X

    def fit_transform(self, X, y=None, **kwargs):
        X = self._construct_encoder(X)
        return self._construct_output(self.encoder.fit_transform(X))

    def fit(self, X, y=None, **kwargs):
        self.fit_transform(X, y=y, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        multi_cat = True if type(self.encoder) in self.__encoders__['multi_categorical'].values() else False
        X = self._reshape_inputs(X, multi_cat)
        return self._construct_output(self.encoder.transform(X))

    @property
    def feature_names(self):
        if self.encoder is None:
            raise NotFittedError(f'Estimator is not yet fitted. Call "fit" or "fit_transform" methods first.')

        if isinstance(self.encoder, (LabelBinarizer, MultiLabelBinarizer)):
            fnames = self.encoder.classes_
        elif isinstance(self.encoder, CountVectorizer):
            fnames = self.encoder.get_feature_names()
        elif isinstance(self.encoder, OneHotEncoder):
            fnames = self.encoder.categories_[0]
        elif isinstance(self.encoder, FeatureHasher):
            fnames = range(self.encoder.n_features)
        else:
            raise NotImplementedError()

        return np.asarray(fnames)

OneHotEncoder(sparse=True)
LabelBinarizer(sparse_output=False)

MultiLabelBinarizer(sparse_output=False)
CountVectorizer()


class BaseEncoderConstructor:
    __encoders__ = {'continuous': __CONT_ENCODERS__,
                    'categorical': CategoricalEncoder,
                    'ordinal': __ORD_ENCODERS__}

    __all_encoders__ = __encoders__.copy()
    __all_encoders__.pop('categorical')
    __all_encoders__ = {k: set(v) for k, v in {**__encoders__['categorical'].__encoders__, **__all_encoders__}.items()}

    __default_enc_params__ = {'continuous': {},
                              'categorical': {'handle_unknown': 'ignore'},
                              'multi_categorical': {},
                              'ordinal': {}}

    @check_options(cont_enc=tuple(__CONT_ENCODERS__), cat_enc=tuple(__CAT_ENCODERS__),
                   multi_cat_enc=tuple(__MULTI_CAT_ENCODERS__))
    @check_types(cont_enc=str, cat_enc=str, multi_cat_enc=str, enc_params=(dict, NoneType), std_categoricals=bool,
                 handle_missing=bool)
    def __init__(self, cont_enc='StandardScaler', cat_enc='OneHotEncoder', multi_cat_enc='MultiLabelBinarizer',
                 enc_params=None, std_categoricals=False, handle_missing=False):

        self.__chosen__ = {'continuous': cont_enc, 'categorical': cat_enc, 'multi_categorical': multi_cat_enc,
                           'ordinal': 'OrdinalEncoder'}
        self.__compute_weights__ = False
        self.__handle_missing__ = handle_missing
        self.__std_categoricals__ = std_categoricals

        self._enc_params = {}
        self._encoders = dict()

        self.set_encoder_params(enc_params)

    @property
    def _chosen(self):
        return self.__chosen__

    @property
    def _std_categoricals(self):
        return self.__std_categoricals__

    @_std_categoricals.setter
    @check_types(flag=bool)
    def _std_categoricals(self, flag):
        self.__std_categoricals__ = flag
        self._set_encoders()  # reset encoders

    @property
    def _handle_missing(self):
        return self.__handle_missing__

    @_handle_missing.setter
    @check_types(flag=bool)
    def _handle_missing(self, flag):
        self.__handle_missing__ = flag
        self._set_encoders()  # reset encoders

    @check_types(enc_params=(dict, NoneType))
    def set_encoder_params(self, enc_params):
        enc_params = enc_params or {}
        for enc_type, params in self.__default_enc_params__.items():
            assert isinstance(params, dict), 'Parameters for each "feature_type" must be passed as a dictionary.'
            self._enc_params[enc_type] = dict(ChainMap(enc_params.get(enc_type) or {},
                                                       self._enc_params.get(enc_type) or {},
                                                       self.__default_enc_params__[enc_type]))
        self._set_encoders()

    def _set_encoders(self):
        # for encoder in set(self.__encoders__).difference(['multi_categorical']):
        for encoder in self.__encoders__:
            self._set_single_encoder(encoder)

    @check_types(params=(NoneType, dict))
    def set_encoder(self, encoder, feature_type, params=None):
        assert feature_type in self.__all_encoders__, f'"feature_type" must be one of {self.__all_encoders__}.'
        assert encoder in self.__all_encoders__[feature_type], f'"encoder" of type {feature_type} must be one of' \
                                                               f' {self.__all_encoders__[feature_type]}.'

        if params:
            self.set_encoder_params({feature_type: params})
        else:
            self._set_encoders()
        self.__chosen__[feature_type] = encoder

    def _set_single_encoder(self, enc_type):
        enc_name = self._chosen[enc_type]

        steps = list()
        # First Step add or not Imputer
        if self._handle_missing:
            steps.append(('Imputer', self._set_imputer(enc_type)))

        # Second Step add correct encoder
        if enc_type == 'categorical':
            steps.append((enc_name, CategoricalEncoder(self._chosen['categorical'],
                                                       self._chosen['multi_categorical'],
                                                       self._enc_params,
                                                       sparse=(not self._std_categoricals))))
        else:
            encoder = self.__encoders__[enc_type][enc_name]
            steps.append((enc_name, encoder(**self._enc_params[enc_type])))

        # Third Step - Adjustments to specific encoders
        # If ordinal encoder no need for imputer (worse since it can't take always string or int), so remove it and
        # add a continuous encoder to set the ordinal to same feature space as the continuous
        cont_enc_name = self._chosen['continuous']
        if enc_type == 'ordinal':
            steps.pop(0)
            steps.append((cont_enc_name,
                          self.__encoders__['continuous'][cont_enc_name](**self._enc_params['continuous'])))

        # if categorical check if should standardize it and if so, how.
        if enc_type in ('categorical', 'multi_categorical'):
            if self._std_categoricals:
                if cont_enc_name != 'StandardScaler':
                    self.__compute_weights__ = True
                else:
                    steps.append(('Standardizer',
                                  self.__encoders__['continuous'][cont_enc_name](**self._enc_params['continuous'])))

        # If only one step return the single encoder else return the pipeline
        self._encoders[enc_type] = steps[0][1] if len(steps) == 1 else Pipeline(steps)

    @staticmethod
    def _set_imputer(enc_type):
        # todo add_indicator flag
        if enc_type == 'continuous':
            imputer = SimpleImputer(strategy='median')
        else:
            # elif enc_type in ('categorical', 'multi_categorical', 'ordinal'):
            fill_value = 'unknown' if enc_type != 'multi_categorical' else ['unknown']
            imputer = SimpleImputer(strategy='constant', fill_value=fill_value)

        return imputer


class Encoder(BaseEncoderConstructor):
    @check_options(return_as=('sparse', 'dataframe', 'array'))
    @check_types(weights=(dict, NoneType), return_as=str, verbose=(int, bool))
    def __init__(self, cont_enc='StandardScaler', cat_enc='OneHotEncoder', multi_cat_enc='MultiLabelBinarizer',
                 enc_params=None, weights=None, std_categoricals=False, handle_missing=False, return_as='sparse',
                 n_jobs=1, verbose=0):
        super().__init__(cont_enc, cat_enc, multi_cat_enc, enc_params, std_categoricals, handle_missing)
        self.encoded_feature_names = None
        self.encoder = None
        self.encoder_y = None
        self.input_features = None
        self.feature_types = dict()
        self.fitted = False
        self.weights = weights

        self._exclude = None
        self._feature_types = None
        self._n_jobs = n_jobs
        self._remainder = None
        self._return_as = return_as
        self._verbose = verbose
        self._weights = None

    def _construct_encoder(self):
        feature_types = {feat: type_ for type_, feats in self.feature_types.items() for feat in feats}
        transformers = [(n, self._encoders[feature_types[n]], [n]) for n in self.input_features
                        if n not in self._exclude]
        sparse = 1 if self._return_as == 'sparse' else 0
        self.encoder = ColumnTransformer(transformers, self._remainder, sparse_threshold=sparse, n_jobs=self._n_jobs,
                                         verbose=self._verbose)

    def _get_features(self, arr):
        if isinstance(arr, Series):
            arr = arr.to_frame(arr.name)

        if isinstance(arr, DataFrame):
            self.input_features = np.asarray(arr.columns, dtype=object)
            assert self.input_features.shape[0] == len(set(self.input_features)), 'There are duplicated features in ' \
                                                                                  'the input.'
        else:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            self.input_features = np.arange(arr.shape[1])
            arr = pd.DataFrame(arr, columns=self.input_features).infer_objects()

        if self._exclude is not None:
            _not_in_features = self._exclude[~np.in1d(self._exclude, self.input_features)]
            if _not_in_features.size > 0:
                self._exclude = self._exclude[~np.in1d(self._exclude, _not_in_features)]
                warnings.warn(f'Features/indexes {_not_in_features} in exclusion list do not exist in the input.'
                              f'Exclusion list was updated.', stacklevel=2)

            self.input_features = self.input_features[~np.in1d(self.input_features, self._exclude)]
            if self._remainder == 'passthrough':
                self.input_features = np.append(self.input_features, self._exclude)

        return arr[self.input_features].copy()

    def _validate_feature_types(self, types):
        if types:
            __accepted_types__ = ('continuous', 'categorical', 'ordinal')
            assert all(key in __accepted_types__ for key in types.keys()), f'"feature_types" keys can only be ' \
                                                                           f'{__accepted_types__}'
            assert all(isinstance(value, (list, tuple)) for value in types.values()), f'"feature_types" values must ' \
                                                                                      f'be either a list or a tuple'

        self._feature_types = types

    def _validate_return_as(self):
        if (self._remainder == 'passthrough') and (self._return_as == 'sparse'):
            if set(self._exclude).intersection(self.feature_types['categorical']):
                warnings.warn('Cannot return sparse output with non-numeric columns passed in exclude as '
                              '"passthrough". Will return as dense array.', stacklevel=2)
            self._return_as = 'array'

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

    def _get_feature_types(self, df):
        __mapped_types__ = {'continuous': 'float', 'categorical': 'str', 'ordinal': 'category'}
        __types_mapping__ = {'str': 'O'}
        __accepted_dtypes__ = ('O', 'category', 'float', 'int')
        self._feature_types_sanity_check(df, __accepted_dtypes__)

        # convert features registered if needed
        for dtype, cols in self._feature_types.items():
            new_type = __mapped_types__[dtype]
            _cols = list(df.loc[:, cols].select_dtypes(exclude=[__types_mapping__.get(new_type, new_type)]).columns)
            if len(_cols) > 0:
                df.loc[:, _cols] = df.loc[:, _cols].astype(new_type)

        for col in df.select_dtypes(include='category').columns:
            df[col].cat.reorder_categories(df[col].dtype.categories, ordered=True, inplace=True)

        # save the feature types
        self.feature_types = {'continuous': list(df.select_dtypes(include=['int', 'float']).columns),
                              'categorical': list(df.select_dtypes(include='O').columns),
                              'ordinal': list(df.select_dtypes(include='category').columns)}

    def _get_encoded_feature_names(self):
        if self.encoder is None:
            raise NotFittedError('Estimator was not yet fitted. Call "fit" or "fit_transform" methods first.')

        def get_names(tf):
            if isinstance(tf, CategoricalEncoder):
                return tf.feature_names
            return ['']

        full_names = []
        for name, trans in self.encoder.named_transformers_.items():
            if isinstance(trans, Pipeline):
                names = [get_names(step_trans) for step_trans in trans.named_steps.values()]
            else:
                if name == 'remainder':
                    names = [self._exclude]
                    name = ''
                else:
                    names = get_names(trans)

            names = list(filter(None, chain.from_iterable(names)))
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
        X[:, weights != 1] *= weights[weights != 1]
        return X

    def _construct_output(self, X):
        # todo
        return X

    @check_options(remainder=('drop', 'passthrough'))
    @check_types(X=(DataFrame, Series, np.ndarray), y=(Series, np.ndarray, NoneType), exclude=(Iterable, NoneType),
                 feature_types=(dict, NoneType))
    def fit_transform(self, X, y=None, feature_types=None, exclude=None, remainder='drop'):
        """

        :param X:
        :return:
        """

        self._validate_feature_types(feature_types)
        self._exclude = np.asarray(exclude, dtype=object) if exclude else None
        self._remainder = remainder

        X = self._get_features(X)
        self._get_feature_types(X)
        self._validate_return_as()

        self._construct_encoder()

        result = self.encoder.fit_transform(X)
        self._get_encoded_feature_names()

        self._calculate_weights(result)
        result = self._apply_weights(result)

        result = self._construct_output(result)
        self.fitted = True

        if y is not None:
            ...
            # construct y_encoder
            return result, y

        return result

    def transform(self, X, y=None):
        # check_is_fitted(self)

        """
        Check if DataFrame or Array;

        retrieve names if dataframe and compare with original names (validate_features) also reorder if needed
        raising a warning

        Column Transform X and y
        construct requested output

        """

        if y is not None:
            ...
            # transform y
            return X, y

        return X

    def fit(self, X, y=None):
        self.fit_transform(X, y=y)
        return self


if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    X.iloc[:, :] = np.where(X.isnull(), np.nan, X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    enc = Encoder(handle_missing=True, return_as='sparse', std_categoricals=True, cont_enc='MinMaxScaler',
                  weights={'pclass': 3, 'age': 5})
    exclude = ['name', 'ticket', 'cabin', 'boat', 'home.dest']
    feature_types = {'continuous': [],
                     'categorical': [],
                     'ordinal': ['pclass']}

    enc.fit_transform(X_train, exclude=exclude, remainder='passthrough', feature_types=feature_types)
