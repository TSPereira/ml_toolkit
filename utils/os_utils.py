import os
import shutil
import math
from collections.abc import Iterable
from uuid import UUID, uuid4
from decorator import decorator


def check_types(**types):
    """Decorator to check types of arguments for function

    :param types: named variables and types to check against. Multiple types for a variable can be passed as tuples
    of types
    :return: function
    """

    @decorator
    def _check(f, *args, **kwds):
        _vars = {**dict(zip(f.__code__.co_varnames, args)), **kwds}
        for var in set(_vars.keys()).intersection(types.keys()):
            # noinspection PyTypeHints
            if not isinstance(_vars[var], types[var]):
                _names = types[var].__name__ if not isinstance(types[var], Iterable) \
                    else tuple(t.__name__ for t in types[var])
                raise TypeError(f'variable {var} must be of type(s): {_names}.')

        return f(*args, **kwds)

    return _check


def check_options(**options):
    """Decorator to check the arguments for function against a pool of options per argument

    :param options: named variables and options to check against. Multiple types for a variable can be passed as tuples
    of types
    :return: function
    """

    @decorator
    def _check(f, *args, **kwds):
        _vars = {**dict(zip(f.__code__.co_varnames, args)), **kwds}
        for var in set(_vars.keys()).intersection(options.keys()):
            assert _vars[var] in options[var], \
                f'variable {var} must be one of {options[var]}.'

        return f(*args, **kwds)

    return _check


def append_to_filename(filename, to_append, ext='', sep='_', remove_ext=False):
    """Function to append a string to a filename passed and optionally and extension
    Output format is <filename><sep><to_append><ext>
    If the filename already contains an extension and this is different from "ext" then output format is
    <filename><sep><to_append><current_extension><ext>

    :param string filename: original filename
    :param string to_append: string to append to the filename passed
    :param string ext: extension to give to the filename.
    :param string sep: separator to use to separate "filename" from "to_append"
    :param bool remove_ext: whether to remove the extension completely
    :return string: new filename
    """

    ext = ('.' + ext.strip('.')) if ext != '' else ext
    curr_name, curr_ext = os.path.splitext(filename)
    name = curr_name + sep + to_append
    ext = (curr_ext + (ext if curr_ext != ext else '')) if not remove_ext else ext
    return name + ext


def is_valid_uuid(uuid_to_test, version=4):
    """Check if uuid_to_test is a valid UUID.

    :param string uuid_to_test: string to test
    :param int version: default: 4. version of uuid to test for. Available versions: {1, 2, 3, 4}
    :return: `True` if uuid_to_test is a valid UUID, otherwise `False`.
    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False

    return str(uuid_obj) == uuid_to_test


@check_types(n=int)
def generate_uuids(n):
    if n < 1:
        raise ValueError('Number of uuids to generate must be > 0')
    return list(str(uuid4()) for _ in range(n))


def version_control(path, bump, bump_prod):
    """Performs the version control of models

    :param str path: Path to the main folder of model_versions
    :param bool bump: Whether to raise the main development model version
    :param bool bump_prod: Whether to raise the main production model version
    :return: Next version path, Current main version
    """

    _current_version = 'v' + str(max([float(f[1:]) for f in os.listdir(path)], default=0.1))

    if bump | bump_prod:
        _current_version = bump_version(path, production_version=bump_prod)
    _version_path = create_version_folder(_current_version, path)

    return _version_path, _current_version


def create_version_folder(version, model_versions_folder, create_data=False):
    """Creates the folder for the version specified if it doesn't exist. It will also create the first sub_folder or
    if the main folder already exists, will retrieve the last version and create the next version sub_folder

    :param str version: version to create
    :param str model_versions_folder: folder where to create the model_versions
    :param bool create_data: whether to create a "data" subfolder
    :return str: path to the current sub_version created
    """

    # create main folder if doesn't exist
    path = os.path.join(model_versions_folder, version)
    os.makedirs(path, exist_ok=True)

    # create subfolder with next sub_version
    sub_version = '0' if not os.listdir(path) else str(max([int(f[f.rfind('.') + 1:]) for f in os.listdir(path)]) + 1)
    path = os.path.join(path, version + '.' + sub_version)
    os.makedirs(path, exist_ok=True)

    # create subfolder "data"
    if create_data:
        os.makedirs(os.path.join(path, 'data'), exist_ok=True)
    return path


@check_types(hist_keep=int)
def clean_version_folders(main_version, model_versions_folder, sub_versions_to_delete='hist', hist_keep=1):
    """Deletes old versions folders of the models

    :param str main_version: main version to run the cleaning algorithm in
    :param str model_versions_folder: folder where to create the model_versions
    :param str|list sub_versions_to_delete: default: 'hist'. Supported: {'hist', 'all', <list of versions>}
    :param int hist_keep: number of versions to keep if "sub_versions_to_delete" == "hist"
    :return: None
    """

    path = os.path.join(model_versions_folder, main_version)
    fld_to_del = os.listdir(path)

    if isinstance(sub_versions_to_delete, str) & (sub_versions_to_delete not in ('hist', 'all')):
        sub_versions_to_delete = [sub_versions_to_delete]

    if sub_versions_to_delete == 'hist':
        if not hist_keep > 0:
            print('Number of versions to keep <=0! Did not delete anything.')
            return

        to_keep = sorted((int(f[f.rfind('.') + 1:]) for f in os.listdir(path)), reverse=True)
        to_keep = [main_version + '.' + str(vers) for vers in to_keep[:hist_keep]]
        for vers in to_keep:
            fld_to_del.remove(vers)

    elif isinstance(sub_versions_to_delete, list):
        fld_to_del = set(sub_versions_to_delete).intersection(fld_to_del)

    for folder in fld_to_del:
        shutil.rmtree(os.path.join(path, folder), ignore_errors=True)

    if not os.listdir(path):
        shutil.rmtree(path, ignore_errors=True)

    return


def bump_version(path, production_version=False):
    """Bumps the main_version of the model

    :param str path: path to the main folder of the model versions
    :param bool production_version: Whether to bump a production version or a dev version. default: False
    :return string: next version denomination of the model
    """

    current_version = max([float(f[1:]) for f in os.listdir(path)])

    if production_version:
        next_version = float(math.ceil(current_version))
    else:
        next_version = current_version + 0.1
    next_version = 'v' + str(next_version)

    os.makedirs(os.path.join(path, next_version), exist_ok=True)
    return next_version


@check_types(files_and_dest=dict)
def copy_files(files_and_dest):
    """
    Copy files to a destination
    :param dict files_and_dest: dictionary with key as full path to file and value as destination folder
    :return:
    """
    for orig, dst in files_and_dest.items():
        shutil.copy(orig, dst)


def copy_files_in_folder(src, dst, file_begins_with=None, extension=None):
    """Copy files from one folder to another - does not copy folders

    :param string src: source path
    :param string dst: destination path
    :param string file_begins_with: copy only files beginning with string passed. For more than one pattern pass a
    tuple. eg: ('model', 'data') will copy any files started with any of the passed strings
    :param string extension: copy only files with passed extension. For more than one extension pass a tuple.
                                eg: ('.csv', '.txt')
    :return:
    """

    src_files = os.listdir(src)

    if file_begins_with:
        assert isinstance(file_begins_with, (str, tuple))
        src_files = [file for file in src_files if file.startswith(file_begins_with)]

    if extension:
        assert isinstance(extension, (str, tuple))
        src_files = [file for file in src_files if file.endswith(extension)]

    for file in src_files:
        full_file_name = os.path.join(src, file)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst)

    return
