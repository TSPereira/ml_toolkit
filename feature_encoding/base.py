from itertools import chain
from collections import ChainMap
from typing import Optional
import warnings

import numpy as np
import scipy.sparse as sp_sparse
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, MinMaxScaler, OrdinalEncoder, \
    StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.exceptions import NotFittedError

try:
    from ..utils.os_utl import check_options, check_types
except (ImportError, ValueError):
    from utils.os_utl import check_options, check_types
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
__ALL_IMPLEMENTED_ENCODERS__ = {**__CONT_ENCODERS__, **__ORD_ENCODERS__, **__CAT_ENCODERS__, **__MULTI_CAT_ENCODERS__}


# todo create callables for most common y_transformations: log, boolean
# todo check how set_params of BaseEstimator work! Override it to work here and in BaseEncoder
class CategoricalEncoder(BaseEstimator):
    """
    todo
    """

    __encoders__ = {'categorical': __CAT_ENCODERS__,
                    'multi_categorical': __MULTI_CAT_ENCODERS__}
    # todo define default enc params per encoder as global variable
    __default_enc_params__ = {'categorical': {'handle_unknown': 'ignore'},
                              'multi_categorical': {'lowercase': False, 'analyzer': list,  # CountVectorizer
                                                    'input_type': 'string'}}               # FeatureHashing

    def __init__(self,
                 cat_enc: str = 'OneHotEncoder',
                 multi_cat_enc: str = 'MultiLabelBinarizer',
                 enc_params: Optional[dict] = None,
                 sparse: bool = True) -> None:

        # todo filter enc_params to only have 'categorical' and 'multi_categorical'
        #  define default params
        self.__chosen__ = {'categorical': cat_enc, 'multi_categorical': multi_cat_enc}

        self.encoder = None
        self._sparse = sparse
        self._enc_params = enc_params or {}
        self._type = None

    def get_params(self, deep: bool = False) -> dict:
        """Returns the parameters of this class

        :param deep: todo
        :return:
        """
        return dict(cat_enc=self._chosen['categorical'], multi_cat_enc=self._chosen['multi_categorical'],
                    sparse=self._sparse, enc_params=self._enc_params)

    @property
    def _chosen(self):
        return self.__chosen__

    @check_types(params=(NoneType, dict), encoder=str, feature_type=str)
    def set_encoder(self,
                    encoder: str,
                    feature_type: str,
                    params: Optional[dict] = None) -> None:
        """Defines one encoder (from the implemented ones) to be used for a specific feature_type (categorical or
        multi_categorical)

        :param encoder: String to choose the encoder from the implemented ones
        :param feature_type: String to assign the chosen encoder to a specific feature_type
        :param params: dictionary with parameters for the encoder passed
        :return:
        """
        assert feature_type in self.__encoders__, f'"feature_type" must be one of {list(self.__encoders__.keys())}.'
        assert encoder in self.__encoders__[feature_type], f'"encoder" of type {feature_type} must be one of ' \
                                                           f'{self.__encoders__[feature_type].keys()}.'

        self.set_encoder_params({feature_type: params or {}})
        self.__chosen__[feature_type] = encoder

    @check_types(enc_params=(dict, NoneType))
    def set_encoder_params(self, enc_params):
        """
        todo to change for set_params once
        :param enc_params:
        :return:
        """

        enc_params = enc_params or {}
        for enc_type, params in self.__default_enc_params__.items():
            assert isinstance(params, dict), 'Parameters for each "feature_type" must be passed as a dictionary.'
            self._enc_params[enc_type] = dict(ChainMap(enc_params.get(enc_type) or {},
                                                       self._enc_params.get(enc_type) or {},
                                                       self.__default_enc_params__.get(enc_type) or {}))

    def _construct_encoder(self, X):
        multi_cat = any(isinstance(val, list) for val in np.asarray(X).reshape(-1, ))
        self._type = 'multi_categorical' if multi_cat else 'categorical'

        # reshape inputs to correct format
        X = self._reshape_inputs(X, multi_cat)

        # get parameters for encoder chosen
        self.set_encoder_params(None)
        params = self._enc_params[self._type]
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
            np.asarray(X, dtype=str).reshape(-1, 1)

    def _construct_output(self, X):
        if self._sparse and not sp_sparse.issparse(X):
            X = sp_sparse.csr_matrix(X)
        elif not self._sparse and sp_sparse.issparse(X):
            X = X.toarray()
        return X

    def fit_transform(self, X, y: Optional = None, **kwargs):
        # todo complete type hinting
        """Fits the encoder to X and outputs the transformation of X

        :param X:
        :param y: Not used. only included for sklearn compatibility
        :param kwargs: Additional fit kwargs. Not used.
        :return:
        """
        X = self._construct_encoder(X)
        return self._construct_output(self.encoder.fit_transform(X))

    def fit(self, X, y: Optional = None, **kwargs) -> object:
        """Fits the encoder to X

        :param X:
        :param y: Not used. only included for sklearn compatibility
        :param kwargs: Additional fit kwargs. Not used.
        :return: instance
        """
        self.fit_transform(X, y=y, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        """Transforms X with previously fitted encoder.

        :param X:
        :param y: Not used. only included for sklearn compatibility
        :param kwargs: Additional fit kwargs. Not used.
        :return:
        """
        multi_cat = True if type(self.encoder) in self.__encoders__['multi_categorical'].values() else False
        X = self._reshape_inputs(X, multi_cat)
        return self._construct_output(self.encoder.transform(X))

    def inverse_transform(self, X):
        """If encoder used provides an inverse transform, apply it to X

        :param X:
        :return:
        """
        try:
            return self.encoder.inverse_transform(X)
        except AttributeError:
            raise NotImplementedError(f'Encoder {self.encoder.__name__} does not implement "inverse_transform" method.')

    @property
    def feature_names(self) -> np.ndarray:
        """Depending on the encoder chosen retrieves the feature names
        :return:
        """
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


__ALL_IMPLEMENTED_ENCODERS__['CategoricalEncoder'] = CategoricalEncoder


class EncoderMixin:
    """
    Todo
    """

    __encoders__ = {'continuous': __CONT_ENCODERS__,
                    'categorical': CategoricalEncoder,
                    'ordinal': __ORD_ENCODERS__}

    __all_encoders__ = __encoders__.copy()
    __all_encoders__.pop('categorical')
    __all_encoders__ = {k: set(v) for k, v in {**__encoders__['categorical'].__encoders__, **__all_encoders__}.items()}

    __default_enc_params__ = {'continuous': {},
                              'categorical': {'handle_unknown': 'ignore'},
                              'multi_categorical': {'lowercase': False, 'analyzer': list,  # CountVectorizer
                                                    'input_type': 'string'},               # FeatureHashing,
                              'ordinal': {}}

    @check_options(cont_enc=tuple(__CONT_ENCODERS__), cat_enc=tuple(__CAT_ENCODERS__),
                   multi_cat_enc=tuple(__MULTI_CAT_ENCODERS__))
    @check_types(cont_enc=str, cat_enc=str, multi_cat_enc=str, enc_params=(dict, NoneType), std_categoricals=bool,
                 handle_missing=bool)
    def __init__(self,
                 cont_enc: str = 'StandardScaler',
                 cat_enc: str = 'OneHotEncoder',
                 multi_cat_enc: str = 'MultiLabelBinarizer',
                 enc_params: Optional[dict] = None,
                 std_categoricals: bool = False,
                 handle_missing: bool = False) -> None:

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

    def _set_encoders(self):
        for enc_type in self.__encoders__:
            enc_name = self._chosen[enc_type]

            steps = list()
            # First Step add or not Imputer
            if self._handle_missing:
                steps.append(('Imputer', self._set_imputer(enc_type)))

            # Second Step add correct encoder
            if enc_type == 'categorical':
                # todo create dict with params for categorical encoder
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

            # Finally, cleanup
            # If only one step return the single encoder else return the pipeline
            self._encoders[enc_type] = steps[0][1] if len(steps) == 1 else Pipeline(steps)

    @staticmethod
    def _set_imputer(enc_type):
        # todo add_indicator flag
        if enc_type == 'continuous':
            imputer = SimpleImputer(strategy='median')
        else:
            # elif enc_type in ('categorical', 'multi_categorical', 'ordinal'):
            imputer = SimpleImputer(strategy='constant', fill_value='unknown')

        return imputer

    @check_types(params=(NoneType, dict))
    def set_encoder(self, encoder: str, feature_type: str, params: Optional[dict] = None) -> None:
        """Defines one encoder (from the implemented ones) to be used for a specific feature_type (categorical or
        multi_categorical)

        :param encoder: String to choose the encoder from the implemented ones
        :param feature_type: String to assign the chosen encoder to a specific feature_type
        :param params: dictionary with parameters for the encoder passed
        :return:
        """
        assert feature_type in self.__all_encoders__, f'"feature_type" must be one of {self.__all_encoders__}.'
        assert encoder in self.__all_encoders__[feature_type], f'"encoder" of type {feature_type} must be one of' \
                                                               f' {self.__all_encoders__[feature_type]}.'

        self.set_encoder_params({feature_type: params | {}})
        self.__chosen__[feature_type] = encoder

    @check_types(enc_params=(dict, NoneType))
    def set_encoder_params(self, enc_params: Optional[dict]):
        """
        todo to change for set_params once
        :param enc_params:
        :return:
        """

        enc_params = enc_params or {}
        for enc_type, params in self.__default_enc_params__.items():
            assert isinstance(params, dict), 'Parameters for each "feature_type" must be passed as a dictionary.'
            self._enc_params[enc_type] = dict(ChainMap(enc_params.get(enc_type) or {},
                                                       self._enc_params.get(enc_type) or {},
                                                       self.__default_enc_params__[enc_type]))
        self._set_encoders()
