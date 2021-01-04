#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from pkg_resources import parse_requirements
from shutil import rmtree
from setuptools import find_packages, setup, Command


here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
with open(os.path.join(here, 'requirements.txt'), 'r') as f:
    REQUIRED = [str(req) for req in parse_requirements(f) if req.name != 'python']
print(REQUIRED)

# What packages are optional?
EXTRAS = {}  # 'fancy feature': ['django']
extras = ['all', 'clustering', 'db_interaction', 'deep_learning', 'feature_encoding', 'time_series',
          'webtools']
for extra in extras:
    rel_path = 'ml_toolkit/requirements.txt' if extra == 'all' else f'ml_toolkit/{extra}/requirements.txt'
    path = os.path.join(here, rel_path)
    with open(path, 'r') as f:
        EXTRAS[extra] = [str(req) for req in parse_requirements(f) if str(req) not in REQUIRED]
print(EXTRAS)

# load metadata from __version__.py file
about = {}
with open(os.path.join(here, 'ml_toolkit', '__version__.py'), 'r') as f:
    for row in f:
        key, value = row.split('=')
        about[key.strip()] = eval(value.strip('\n'))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = about['__description__']


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    python_requires='>=3.7.7',
    url=about['__url__'],
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=about['__license__'],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        # 'upload': UploadCommand,
    },
)
