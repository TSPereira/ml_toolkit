import numpy as np
import pandas as pd
from pandas.core.api import Series, DataFrame
import scipy as sp
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher

from _closer_packages.log_utils import printv, print_progress_bar


class StdEncoder(object):
    """
    Class to hold encoders of features

    Attributes
    ----------
    name: name of the class instance to identify the encoder in later stages
    encoder: type of encoder used for the current feature. eg: MinMaxScaler()
    position: tuple with position of the encoded columns in the final encoded array. eg: (0, 5)
    gamma: value of weight for the feature
    feature_category: list of headers for the encoded array based on the original feature name and the categories in it
    """

    def __init__(self, name, encoder=None, position=None, gamma=1):
        """Class constructor

        :param string name: mandatory, name of the class instance to identify the encoder in later stages
        :param encoder: optional, default: None, type of encoder used for the current feature.
                        eg: MinMaxScaler()
        :param tuple position: optional, default: None, tuple with position of the encoded columns in the final
        encoded array.
                        eg: (0, 5) => 5 columns positioned from 0 to 5 (exclusive) of array
        :param float gamma: optional, default: 1, value of weight for the feature
        """

        self.name = name
        self.encoder = encoder
        self.position = position
        self.gamma = gamma
        self.feature_category = list()

    def get_feature_category(self, sep=' | '):
        """Define the headers on the encoded array for this feature based on original feature name and categories

        :param string sep: separator to use between feature name and feature categories names
        :return: None
        """

        # If the feature was expanded into more than one column, create the new names of each column
        if self.position[1]-self.position[0] > 1:
            for i in range(self.position[1]-self.position[0]):
                self.feature_category.append(self.name + sep + str(self.encoder.classes_[i]))
        else:
            self.feature_category.append(self.name)


def _category_encode(column, multi_category_transformer='CountVectorizer', **kwargs):
    """Function to encode categorical features
    If column passed is filled with lists it will use CountVectorizer, MultiLabelBinarizer or FeatureHashing.
    Else it will use LabelBinarizer

    :param pandas.Series column: DataFrame column to be encoded
    :param string multi_category_transformer: default: 'CountVectorizer', One of ['CountVectorizer',
    'MultiLabelBinarizer', 'FeatureHashing']
    :return: col_ohe: Sparse Array with the column encoded
    :return: enc: Encoder for this column.
    """

    assert multi_category_transformer in ['CountVectorizer', 'MultiLabelBinarizer', 'FeatureHashing'], \
        "'multi_category_transformer must be one of ['CountVectorizer', 'MultiLabelBinarizer', 'FeatureHashing']"

    # Copy to avoid direct changes in the original dataframe
    _col = column.copy(deep=True)
    _types = _col.apply(type)

    # If the column is composed of lists
    if list in _col.apply(type).values:
        # Convert any non-list element in list
        _col[_col.apply(type) == list] = _col[_col.apply(type) == list].apply(lambda x: [str(x_) for x_ in x])
        _col[_col.apply(type) != list] = _col[_col.apply(type) != list].apply(lambda x: [str(x)])
        enc, col_ohe = None, np.array([])

        if multi_category_transformer == 'CountVectorizer':
            enc = CountVectorizer(lowercase=False, analyzer=list)
            col_ohe = enc.fit_transform(_col)
            enc.classes_ = enc.get_feature_names()

        elif multi_category_transformer == 'MultiLabelBinarizer':
            enc = MultiLabelBinarizer()
            col_ohe = enc.fit_transform(_col)

        elif multi_category_transformer == 'FeatureHashing':
            assert 'n_features' in kwargs, "When 'multi_category_transformer' == 'FeatureHashing' it is necessary to " \
                                           "pass the 'n_features' to reduce the categories to."
            enc = FeatureHasher(n_features=kwargs['n_features'], input_type='string')
            col_ohe = enc.fit_transform(_col)
            enc.classes_ = [str(x) for x in range(kwargs['n_features'])]

    # For other categoricals
    else:
        enc = LabelBinarizer()
        col_ohe = enc.fit_transform(_col.astype(str))

    return sp.sparse.csr_matrix(col_ohe), enc


def _get_encoder_types(df, method='auto'):
    """Define the type of features in the dataframe based on their dtype and depending on the method to use
    Features in the DataFrame should all be of types: {'object', 'category', 'float' or 'int'}

    :param pandas DataFrame df: DataFrame to encode
    :param string method: default: 'auto'. One of: ['auto', 'kmeans', 'minibatchkmeans', 'kmodes', 'kprototypes',
    'DBSCAN']
    :return:
    """

    ord_feat, cont_feat, cat_feat = [], [], []

    if method in ['auto', 'kmeans', 'minibatchkmeans', 'DBSCAN']:
        cat_feat = df.select_dtypes(include='O').columns.tolist()
        ord_feat = df.select_dtypes(include='category').columns.tolist()
        cont_feat = df.select_dtypes(include=['float', 'int', 'float64', 'int64']).columns.tolist()

    elif method in ['kmodes']:
        cat_feat = df.select_dtypes(include='O').columns.tolist()
        ord_feat = df.select_dtypes(include='category').columns.tolist()

        print('KModes selected. Any continuous feature will be binned and converted to an ordinal feature.')
        cont_feat = df.select_dtypes(include=['float', 'int', 'float64', 'int64']).columns.tolist()
        for col in cont_feat:
            df[col] = pd.cut(df[col], 3).astype(str)
        ord_feat += cont_feat
        cont_feat = []

    elif method in ['kprototypes']:
        cont_feat = df.select_dtypes(include='category').columns.tolist() + df.select_dtypes(
            include=['float', 'int', 'float64', 'int64']).columns.tolist()
        cat_feat = df.select_dtypes(include='O').columns.tolist()

    return cat_feat, ord_feat, cont_feat


def _get_gamma(cat_column, cont_table):
    """Calculate the weight that a categorical feature should have so that its influence in the model is the same
    order as continuous features. The weight is calculated using the Std Deviations of the categorical variable and
    all the continuous variables

    :param scipy.sparse.csr_matrix cat_column: categorical column to calculate gamma for
    :param scipy.sparse.csr_matrix cont_table: table with continuous variables encoded
    :return float: value of gamma for cat_column passed
    """

    gamma = 1
    if cont_table.shape[1] > 0 and cat_column.shape[1] > 0:
        # get the mean std deviations for continuous table and max std deviation of categorical tables
        cont_stdev = np.mean(np.std(cont_table.todense(), axis=0))
        other_stdev = np.max(np.std(cat_column.todense(), axis=0))

        gamma = cont_stdev / other_stdev

    return gamma


def transform_features_type(data, variables, order=None):
    """Converts the dtype of the columns in a DataFrame according to the type of feature the user choose that
    column to be.
    Ordinal features will be converted to 'category' dtype
    Categorical features will be converted to 'object' dtype
    Continuous features will be converted to 'float' dtype

    :param pandas.DataFrame data: DataFrame to convert
    :param dict variables: can contain three distinct keys: 'categoricals', 'continuous', 'ordinals'. If none of
    these are present it raises and error. If at least one is present it will perform that conversion and raise a
    warning
    :param dict order: ordinal features order should be passed as a dictionary of format:
    {'feature_name': list(categories_order)}
    :return: pandas DataFrame with corrected dtypes
    """

    types = {'ordinals': 'category', 'categoricals': 'str', 'continuous': 'float'}

    # sanity checks
    assert isinstance(data, (DataFrame, Series)), \
        f'"data" must be of type "Pandas DataFrame or Series".'
    assert isinstance(variables, dict), f'"variables" must be passed as a dictionary'
    assert all(True if isinstance(value, (list, str)) else False for value in variables.values()), \
        f'Values of "variables" must be either of type "list" or "str".'
    assert isinstance(order, (type(None), dict)), f'"order"" must be either "None" or of type "dict".'
    _check_keys = [key for key in variables.keys() if key in types.keys()]
    assert _check_keys, f'None of the keys passed ({variables.keys()}) are valid! Pass one of {types.keys()}'

    df = data.copy(deep=True)
    # if Series convert to correct format as DataFrame
    if isinstance(df, Series):
        df = df.to_frame().transpose()

    # convert columns
    for key in _check_keys:
        cols = variables[key]
        if isinstance(cols, list) and (len(cols) == 1):
            cols = cols[0]
        df.loc[:, cols] = df.loc[:, cols].astype(types[key])

    # for columns that are categories, assign an order
    for col in df.select_dtypes(include='category').columns:
        if order is not None:
            _categories = order[col] if col in order else df[col].dtype.categories
        else:
            _categories = df[col].dtype.categories

        df[col].cat.reorder_categories(_categories, ordered=True)

    return df


def encode_training(data, method='auto', cont_transformer='MinMaxScaler', gamma=None, calc_gamma=False,
                    multi_cat_transformer='MultiLabelBinarizer', return_as='sparse', verbose=0, **kwargs):
    # todo include feature range for MinMaxScaler: cont_range = (0, 1),
    #  include RobustScaler and PowerTransformer for continuous normalization
    """
    It encodes the information in a DataFrame so that all columns are represented in the correct form to be fed
    to a clustering algorithm
    :param pandas.DataFrame data: data to encode
    :param string method: default='auto' method to be used from ['auto', 'kmeans', 'minibatchkmeans', 'kmodes',
    'kprototypes', 'DBSCAN']
    :param string cont_transformer: default: 'MinMaxScaler'. One of ['MinMaxScaler', 'StandardScaler']. How numerical
    features should be encoded.
    :param dict gamma: default=None, dictionary with passed weights for specific features. Features not existing in the
    dictionary will either be assigned a weight of 1, or have their weight calculated (if calc_gamma=True)
    :param bool calc_gamma: default: False, whether it should calculate the correct weight to use on categorical
    features
    :param string multi_cat_transformer: default: 'CountVectorizer'. One of 'CountVectorizer', 'MultiLabelBinarizer' or
    'FeatureHashing'. How features composed of lists should be encoded.
    :param int verbose: default: 0, parameter to control verbosity
    :return scipe.sparse.csr_matrix, list: returns a sparse array of encoded data and a list of the encoders used
    (of class StdEncoder)
    """

    __cont_transformers__ = {'MinMaxScaler': MinMaxScaler,
                             'StandardScaler': StandardScaler}  # , 'RobustScaler', 'PowerTransformer']
    __cat_transformers__ = ['CountVectorizer', 'MultiLabelBinarizer', 'FeatureHashing']

    # sanity checks
    if gamma is not None:
        assert type(gamma) is dict
    else:
        gamma = dict()

    assert cont_transformer in __cont_transformers__.keys()
    assert multi_cat_transformer in __cat_transformers__
    assert return_as in ('sparse', 'dataframe')

    # assert (isinstance(cont_range, tuple) & (len(cont_range) == 2))
    # if cont_transformer == 'MinMaxScaler':
    #   assert cont_range[1] > cont_range[0]

    df = data.copy(deep=True)

    printv('Encoding features...', verbose=verbose)
    # Initialize sparse arrays with the correct number or rows
    ohe_table = sp.sparse.csr_matrix((data.shape[0], 0))
    cont_table = sp.sparse.csr_matrix((data.shape[0], 0))

    # Filter features by their type
    cat_feat, ord_feat, cont_feat = _get_encoder_types(df, method)

    # initialize lists
    encoders = []
    features = []

    # Normalize continuous variables
    for idx, feat in enumerate(cont_feat):
        print_progress_bar(idx+1, len(cont_feat), prefix='Encoding continuous features: ', suffix=f'| Encoding {feat}',
                           level=2, verbose=verbose)
        scaler = __cont_transformers__[cont_transformer]()

        cont_vect = sp.sparse.csr_matrix(scaler.fit_transform(np.array(df[feat]).reshape(-1, 1)))
        _pos = (ohe_table.shape[1], ohe_table.shape[1] + cont_vect.shape[1])

        if feat in gamma:
            _gamma = gamma[feat]
        else:
            _gamma = 1

        _current_encoder = StdEncoder(str(feat), scaler, _pos, _gamma)
        _current_encoder.get_feature_category()
        encoders.append(_current_encoder)
        printv('gamma: {}'.format(_gamma), level=3, verbose=verbose)

        features += _current_encoder.feature_category
        cont_table = sp.sparse.hstack([cont_table, cont_vect])
        ohe_table = sp.sparse.hstack([ohe_table, cont_vect*_gamma])

    # Label Encode ordinal variables
    for idx, feat in enumerate(ord_feat):
        print_progress_bar(idx+1, len(ord_feat), prefix='Encoding ordinal features: ', suffix=f'| Encoding {feat}',
                           level=2, verbose=verbose)
        le = LabelEncoder()

        cat_vect = sp.sparse.csr_matrix(le.fit_transform(df[feat])).transpose()
        _pos = (ohe_table.shape[1], ohe_table.shape[1] + cat_vect.shape[1])

        if feat in gamma:
            _gamma = gamma[feat]
        elif calc_gamma:
            _gamma = _get_gamma(cat_vect, cont_table)
        else:
            _gamma = 1

        _current_encoder = StdEncoder(str(feat), le, _pos, _gamma)
        _current_encoder.get_feature_category()
        encoders.append(_current_encoder)
        printv('gamma: {}'.format(_gamma), level=3, verbose=verbose)

        features += _current_encoder.feature_category
        ohe_table = sp.sparse.hstack([ohe_table, cat_vect*_gamma])

    for idx, feat in enumerate(cat_feat):
        print_progress_bar(idx+1, len(cat_feat), prefix='Encoding categorical features: ', suffix=f'| Encoding {feat}',
                           level=2, verbose=verbose)

        ohe_matrix, _encoder = _category_encode(df[feat], multi_category_transformer=multi_cat_transformer, **kwargs)
        _pos = (ohe_table.shape[1], ohe_table.shape[1] + ohe_matrix.shape[1])

        if feat in gamma:
            _gamma = gamma[feat]
        elif calc_gamma:
            _gamma = _get_gamma(ohe_matrix, cont_table)
        else:
            _gamma = 1

        _current_encoder = StdEncoder(str(feat), _encoder, _pos, _gamma)
        _current_encoder.get_feature_category()
        encoders.append(_current_encoder)
        printv('gamma: {}'.format(_gamma), level=3, verbose=verbose)

        features += _current_encoder.feature_category
        ohe_table = sp.sparse.hstack([ohe_table, ohe_matrix*_gamma])

    if (return_as == 'dataframe') & isinstance(data, pd.DataFrame):
        ohe_table = pd.DataFrame(ohe_table.toarray(), index=data.index, columns=features)

    return ohe_table, encoders


def encode_predict(data, encoders, ignore_new_categories=False, return_as='array', verbose=0):
    """Function to encode data using encoders from training set

    :param pandas.DataFrame data: data from prediction to encode
    :param list encoders: list of class instances of type StdEncoders with the encoders used on training
    :param bool, ignore_new_categories: default: False, flag to control whether new (unseen during training)
    categories should be ignored or if rows containing them should be ignored from encoding
    :param int, verbose: default: 0, parameter to control verbosity
    :return numpy.ndarray, list: table encoded, list containing the indexes of rows not encoded due to categories not
    seen during training
    """

    to_encode = data.copy(deep=True)
    not_to_encode = []
    dim = max((enc.position[1] for enc in encoders))

    printv('Encoding features...', verbose=verbose)

    if not ignore_new_categories:
        # Find which lines to encode and which have non seen categories and only to be predicted at next training
        printv('Number of lines to be predicted on next training due to each variable', level=2, verbose=verbose)
        _encoders = {enc.name: enc for enc in encoders if not isinstance(enc.encoder, (MinMaxScaler, CountVectorizer))}
        for col in to_encode.columns:
            if col in _encoders:
                enc_classes = _encoders[col].encoder.classes_
                _prev_count = len(not_to_encode)

                not_to_encode += to_encode[~to_encode[col].isin(enc_classes)].index.tolist()
                to_encode = to_encode[to_encode[col].isin(enc_classes)]
                printv('{}, {}'.format(col, len(not_to_encode) - _prev_count), level=2, verbose=verbose)
        printv('', level=2, verbose=verbose)

    # Initialize encoded array with zeros
    ohe_table = np.zeros((to_encode.shape[0], dim))

    _encoders = {enc.name: enc for enc in encoders}
    for idx, col in enumerate(to_encode.columns):
        print_progress_bar(idx + 1, len(to_encode.columns), prefix='Encoding features: ', suffix=f'| Encoding {col}',
                           level=2, verbose=verbose)
        enc = _encoders[col]

        try:
            if not isinstance(enc.encoder, CountVectorizer):
                aux = enc.encoder.transform(np.array(to_encode[col]).reshape(-1, 1))
                ohe_table[:, enc.position[0]:enc.position[1]] = aux.reshape(aux.shape[0], -1) * enc.gamma
            else:
                aux = enc.encoder.transform(to_encode[col])
                ohe_table[:, enc.position[0]:enc.position[1]] = aux.reshape(aux.shape[0], -1).toarray() * enc.gamma

        except IndexError:
            print('No rows left to encode/predict.')
            break

    if (return_as == 'dataframe') & isinstance(data, pd.DataFrame):
        features = [feat for enc in encoders for feat in enc.feature_category]
        ohe_table = pd.DataFrame(ohe_table, index=to_encode.index, columns=features)

    return ohe_table, not_to_encode


# def _convert_encoder_range(orig_enc, cont_range):
#     encoder = deepcopy(orig_enc)
#     encoder.feature_range = cont_range
#     encoder.min_ = encoder.feature_range[0]
#     encoder.scale_ = orig_enc.scale_ * (encoder.feature_range[1]-encoder.feature_range[0]) / \
#                      (orig_enc.feature_range[1] - orig_enc.feature_range[0])
#
#     return encoder
#
#
# def _convert_matrix_range(arr, cont_range):
#     range = cont_range[1] - cont_range[0]
#     return arr * range + sp.sparse.csr_matrix(np.ones(arr.shape) * cont_range[0])
