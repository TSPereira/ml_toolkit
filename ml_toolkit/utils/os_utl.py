import math
import os
import shutil
import warnings
from inspect import signature, getfullargspec
from uuid import UUID, uuid4
from typing import Tuple, Iterable, Callable, Sequence, Any, Optional, Union, List

from decorator import decorator
NoneType = type(None)


def check_types(**types: Union[type, Sequence]) -> Callable:
    """Decorator to check types of arguments for function

    Args:
        **types: named variables and types to check against. Multiple types for a variable can be passed as tuples
                 of types. Variables not in the function signature will have no effect. Variables in the function
                 signature, but not in the decorator arguments will not be checked

    Returns:
        Callable: original function decorated

    Examples:
        >>> @check_types(a=int, b=str)
        >>> def func(a, b):
        >>>      ...

        >>> @check_types(a=(int, str), b=str)
        >>> def func(a, b)
        >>>      ...

        >>> @check_types(a=(int, str), b=str, c=dict)
        >>> def func(a, b, d)
        >>>      ...
    """
    @decorator
    def _check(f, *args, **kwds):
        _vars = {**dict(zip(getfullargspec(f).args, args)), **kwds}
        for var in set(_vars.keys()).intersection(types.keys()):
            # noinspection PyTypeHints
            if not isinstance(_vars[var], types[var]):
                _names = get_type_name(types[var]) if not isinstance(types[var], Iterable) \
                    else tuple(get_type_name(t) for t in types[var])
                raise TypeError(f'variable {var} must be of type(s): {_names}.')

        return f(*args, **kwds)

    return _check


def get_type_name(t: Any) -> str:
    """Find the name of a type passed. It will look in sequence for "__name__" and "_name" and if both fail it will
    take the str(t)

    Args:
        t: Any object

    Returns:
        String: string with the name of the type
    """
    return getattr(t, '__name__', getattr(t, '_name', str(t)))


def check_options(**options: Sequence) -> Callable:
    """Decorator to check the arguments for function against a pool of options per argument

    Args:
        **options: named variables and options to check against. Options for each argument should be a single iterable.
                   Variables not in the function signature will have no effect. Variables in the function signature,
                   but not in the decorator arguments will not be checked

    Returns:
        Callable: original function decorated

    Examples:
        >>> @check_options(a=[1,2,3], b=('a', 'b', 'c'))
        >>> def func(a, b):
        >>>      ...
    """
    @decorator
    def _check(f, *args, **kwds):
        _vars = {**dict(zip(getfullargspec(f).args, args)), **kwds}
        for var in set(_vars.keys()).intersection(options.keys()):
            if _vars[var] not in options[var]:
                raise KeyError(f'variable {var} must be one of {options[var]}.')

        return f(*args, **kwds)

    return _check


@check_options(closed=('min', 'max', 'both', None))
@check_types(min_value=(float, int, NoneType), max_value=(float, int, NoneType))
def check_interval(items: Union[str, Sequence], min_value: Optional[Union[int, float]] = None,
                   max_value: Optional[Union[int, float]] = None, closed: Optional[str] = None) -> Callable:
    """Decorator to check if arguments are inside of a given interval

    Args:
        items: One or more variables of the function to be checked
        min_value: minimum value of the interval
        max_value: maximum value of the interval
        closed: Whether the interval should be considered to be closed on the "min" side, "max" side, "both" sides
                or fully open (None).

    Returns:
        Callable: original function decorated
    """
    items = items if isinstance(items, Sequence) else [items]

    # function to create the message in case of failure
    def gen_ans(name):
        prefix = f'Variable "{name}" must be'
        left_sign = '<=' if closed in ('min', 'both') else '<'
        right_sign = '<=' if closed in ('max', 'both') else '<'

        if (min_value is not None) and (max_value is not None):
            return f'{prefix} between {min_value} {left_sign} {name} {right_sign} {max_value}.'

        elif max_value is not None:
            return f'{prefix} {right_sign} {max_value}'

        else:
            left_sign = f'>{left_sign[1:]}'
            return f'{prefix} {left_sign} {min_value}'

    @decorator
    def _check(f, *args, **kwargs):
        _vars = {**dict(zip(f.__code__.co_varnames, args)), **kwargs}
        for var in set(_vars.keys()).intersection(items):
            x = _vars[var]
            low = True if min_value is None else (x >= min_value if closed in ('min', 'both') else x > min_value)
            high = True if max_value is None else (x <= max_value if closed in ('max', 'both') else x < max_value)
            if not low & high:
                raise ValueError(gen_ans(var))

        return f(*args, **kwargs)

    return _check


def filter_kwargs(d: dict, func: Callable) -> dict:
    """Filter dictionary items to only contain arguments of the func passed

    Args:
        d: dictionary to be filtered
        func: callable (function or method) to get arguments to filter for

    Returns:
        Dict: filtered dictionary

    Examples:
        >>> def some_func(a=1, b=2):
        >>>     return

        >>> filter_kwargs(dict(a=10, c=1000), some_func)
        {'a':10}

        >>> filter_kwargs(dict(a=10, b=100, c=1000), some_func)
        {'a':10, 'b':100}

        >>> filter_kwargs(dict(c=1000), some_func)
        {}
    """
    acceptable = signature(func).parameters.keys()
    return {key: value for key, value in d.items() if key in acceptable}


def append_to_filename(filename: str, to_append: str, ext: str = None, sep: str = '_', remove_ext: bool = False) -> str:
    """Function to append a string to a filename passed and optionally an extension. Output format is
    <filename><sep><to_append><ext>.

    If the filename already contains an extension and this is different from "ext" then output format is
    <filename><sep><to_append><current_extension><ext>

    Args:
        filename: original filename
        to_append: string to append to the filename passed
        ext: extension to give to the filename.
        sep: separator to use to separate "filename" from "to_append"
        remove_ext: whether to remove the extension completely

    Returns:
        String: new filename

    Examples:
        >>> append_to_filename('test.txt', 'option1', '.csv', sep='_', remove_ext=True)
        'test_option1.csv'

        >>> append_to_filename('test.txt', 'option1', '.csv', sep='_', remove_ext=False)
        'test_option1.txt.csv'

        >>> append_to_filename('test.txt', 'option1', ext=None, sep='_', remove_ext=False)
        'test_option1.txt'

        >>> append_to_filename('test.txt', 'option1', ext=None, sep='_', remove_ext=True)
        'test_option1'
    """
    ext = ('.' + ext.strip('.')) if ext is not None else ''
    curr_name, curr_ext = os.path.splitext(filename)
    name = curr_name + sep + to_append
    ext = (curr_ext + (ext if curr_ext != ext else '')) if not remove_ext else ext
    return name + ext


def is_valid_uuid(uuid_to_test: str, version: int = 4) -> bool:
    """Check if uuid_to_test is a valid UUID.

    Args:
        uuid_to_test: string to test
        version: default: 4. version of uuid to test for. Available versions: {1, 2, 3, 4}

    Returns:
        Bool: `True` if uuid_to_test is a valid UUID, otherwise `False`.

    Examples:
        >>> is_valid_uuid('3d1c4bda-faed-44aa-b34a-fa524ec0345a')
        True

        >>> is_valid_uuid('3d1c4bda-faed-44aa-b34a-fa52')
        False
    """
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False

    return str(uuid_obj) == uuid_to_test


def generate_uuids(n: int) -> List[str]:
    """Generates a list of UUIDs (only of version 4) of size "n"

    Args:
        n: number of UUIDs to generate

    Returns:
        List: List of n UUIDs 4

    Examples:
        >>> generate_uuids(5)
        ['fdf5cdb3-0b93-453c-83be-b5179b4145b5',
         '6a93faa0-1e9d-4c93-8a52-7ccffc0ae85a',
         '481d0574-c320-4ef2-a9f2-6e1b4e0c3f9b',
         'f9fb3959-a7c5-45ae-ac62-3f8dea533df3',
         '72edee7b-6c9e-484c-8e7d-c1c0db082c78']
    """
    if n < 1:
        raise ValueError('Number of uuids to generate must be > 0')
    return list(str(uuid4()) for _ in range(n))


def version_control(path: str, bump: bool, bump_prod: bool) -> Tuple[str, str]:
    """Performs the version control of models

    :param str path: Path to the main folder of model_versions
    :param bool bump: Whether to raise the main development model version
    :param bool bump_prod: Whether to raise the main production model version
    :return: Next version path, Current main version
    """

    os.makedirs(path, exist_ok=True)
    if bump | bump_prod:
        _current_version = bump_version(path, production_version=bump_prod)
    else:
        _versions = [f for f in os.listdir(path) if (f.startswith('v') and f[-1].isdigit())]
        _current_version = 'v' + str(max([float(version[1:]) for version in _versions], default=0.1))

    _version_path = create_version_folder(_current_version, path)
    return _version_path, _current_version


@check_types(create_subfolders=Iterable)
def create_version_folder(version: str, model_versions_folder: str, create_subfolders: Iterable = None) -> str:
    """Creates the folder for the version specified if it doesn't exist. It will also create the first sub_folder or
    if the main folder already exists, will retrieve the last version and create the next version sub_folder

    :param str version: version to create
    :param str model_versions_folder: folder where to create the model_versions
    :param Iterable create_subfolders: Iterable containing any subfolders to be created. Nested folders can be
    created by passing "subfolder/nested_subfolder" path in the iterable
    :return str: path to the current sub_version created
    """

    # create main folder if doesn't exist
    path = os.path.join(model_versions_folder, version)
    os.makedirs(path, exist_ok=True)

    # create subfolder with next sub_version
    sub_version = '0' if not os.listdir(path) else str(max([int(f[f.rfind('.') + 1:]) for f in os.listdir(path)]) + 1)
    path = os.path.join(path, version + '.' + sub_version.zfill(3))
    os.makedirs(path, exist_ok=True)

    # create subfolder "data"
    if create_subfolders is not None:
        for subfolder in create_subfolders:
            os.makedirs(os.path.join(path, subfolder), exist_ok=True)
    return path


@check_types(hist_keep=int)
def clean_version_folders(main_version: str, model_versions_folder: str, sub_versions_to_delete: str = 'hist',
                          hist_keep: int = 1) -> None:
    """Deletes old versions folders of the models

    :param str main_version: main version to run the cleaning algorithm in
    :param str model_versions_folder: folder where to create the model_versions
    :param str|list sub_versions_to_delete: default: 'hist'. Supported: {'hist', 'all', <list of versions>}
    :param int hist_keep: number of versions to keep if "sub_versions_to_delete" == "hist"
    :return: None
    """

    path = os.path.join(model_versions_folder, main_version)
    fld_to_del = set(os.listdir(path))

    if isinstance(sub_versions_to_delete, str) & (sub_versions_to_delete not in ('hist', 'all')):
        sub_versions_to_delete = [sub_versions_to_delete]

    if sub_versions_to_delete == 'hist':
        if not hist_keep > 0:
            print('Number of versions to keep <=0! Did not delete anything.')
            return

        to_keep = sorted((int(f[f.rfind('.') + 1:]) for f in os.listdir(path) if
                          (f.startswith('v') and f[-1].isdigit())), reverse=True)
        to_keep = [main_version + '.' + str(vers) for vers in to_keep[:hist_keep]]
        fld_to_del = fld_to_del.difference(to_keep)

    elif isinstance(sub_versions_to_delete, list):
        fld_to_del = set(sub_versions_to_delete).intersection(fld_to_del)

    for folder in fld_to_del:
        shutil.rmtree(os.path.join(path, folder), ignore_errors=True)

    if not os.listdir(path):
        shutil.rmtree(path, ignore_errors=True)

    return


def bump_version(path: str, production_version: bool = False) -> str:
    """Bumps the main_version of the model

    :param str path: path to the main folder of the model versions
    :param bool production_version: Whether to bump a production version or a dev version. default: False
    :return string: next version denomination of the model
    """

    _current_version = max([float(f[1:]) for f in os.listdir(path) if (f.startswith('v') and f[-1].isdigit())],
                           default=0.1)
    if production_version:
        next_version = float(math.ceil(_current_version))
    else:
        next_version = round(_current_version + 0.1, 6)
    next_version = 'v' + str(next_version)

    os.makedirs(os.path.join(path, next_version), exist_ok=True)
    return next_version


def list_files(path: str, prefix: str = '', ext: Optional[str] = None, limit: int = -1,
               nested_first: bool = False, include: Optional[list] = None, exclude: Optional[list] = None) -> List[str]:
    """List all files in a folder and subfolders recursively respecting options

    Args:
        path: path to folder to inspect
        prefix: prefix to filter files by
        ext: extension to filter files by
        limit: max number of files to return
        nested_first: Whether to explore subfolders first or append current level files first
        include: Folders to include in the search
        exclude: Folders to exclude from the search

    Returns:
        List of file paths
    """

    def check_dirs():
        all_files = []
        for d in dirs:
            if d in (exclude or []):
                continue

            nested_files = list_files(os.path.join(root, d), prefix, ext, limit, nested_first, include, exclude)
            if d not in (include or dirs) and not nested_files:
                continue

            all_files.extend(nested_files)
            if 0 < limit <= len(all_files):
                break

        return all_files

    def check_ext(file, extension):
        return True if extension is None else os.path.splitext(file)[1] == ext

    def check_include_exclude(root):
        _dirs = os.path.normpath(root).split(os.sep)
        include_condition = (include is None or any(d in include for d in _dirs))
        exclude_condition = _dirs[-1] not in (exclude or [])
        return include_condition and exclude_condition

    #
    # get current folder dirs and files
    root, dirs, files = next(os.walk(path), (None, [], []))

    # initialization
    new_files = []
    if check_include_exclude(root):
        new_files = [os.path.normpath(os.path.join(root, file)) for file in files
                     if file.startswith(prefix) and check_ext(file, ext)]

    # get nested files
    new_files = (check_dirs() + new_files) if nested_first else (new_files + check_dirs())
    return new_files[:limit] if limit > 0 else new_files


def copy_files(files_and_dest: dict) -> None:
    """
    Copy files to a destination
    :param dict files_and_dest: dictionary with key as full path to file and value as destination folder
    :return:
    """
    for orig, dst in files_and_dest.items():
        shutil.copy(orig, dst)


def move_files_in_folder(src: str, dst: str, file_begins_with: bool = None, extension: bool = None,
                         copy: bool = False, overwrite: bool = False) -> None:
    """Move or copy files from one folder to another - does not copy folders

    :param string src: source path
    :param string dst: destination path
    :param string file_begins_with: copy only files beginning with string passed. For more than one pattern pass a
    tuple. eg: ('model', 'data') will copy any files started with any of the passed strings
    :param string extension: copy only files with passed extension. For more than one extension pass a tuple.
                                eg: ('.csv', '.txt')
    :param boolean copy: Whether to copy the files instead of moving them
    :param boolean overwrite: whether to overwrite files
    :return:
    """

    # define function to use
    func = shutil.copy if copy else shutil.move

    # sanity checks
    if src == dst:
        warnings.warn(f'Source and destination folder are the same ({src}). Nothing moved/copied.', stacklevel=2)
        return

    if not os.path.isdir(src):
        raise FileNotFoundError(f'Source folder ({src}) does not exist.')
    os.makedirs(dst, exist_ok=True)

    src_files = os.listdir(src)
    if not src_files:
        warnings.warn(f'No files in the source folder ({src}). Nothing moved/copied.', stacklevel=2)
        return

    # filters
    if file_begins_with:
        assert isinstance(file_begins_with, (str, tuple))
        src_files = [file for file in src_files if file.startswith(file_begins_with)]

    if extension:
        assert isinstance(extension, (str, tuple))
        src_files = [file for file in src_files if file.endswith(extension)]

    # process
    for file in src_files:
        source_filename = os.path.join(src, file)
        dst_filename = os.path.join(dst, file)
        if os.path.isfile(source_filename):
            if os.path.isfile(dst_filename) and not overwrite:
                warnings.warn(f'Could not {func.__name__} file {source_filename} because there is a file with the '
                              f'same name in the destination folder.\nPass the "overwrite" parameter if you want '
                              f'to overwrite.', stacklevel=2)
                continue

            func(source_filename, dst)

    return

