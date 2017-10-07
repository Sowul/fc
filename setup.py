#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup

NAME = 'fc'
DESCRIPTION = 'Genetic algorithm for feature engineering. '
URL = 'https://github.com/Sowul/fc'
EMAIL = 'bartosz.sowul@protonmail.com'
AUTHOR = 'Bartosz Sowul'

REQUIRED = [
    'matplotlib', 'numpy', 'tqdm', 'scikit_learn',
]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

about = {}
with open(os.path.join(here, '__version__.py')) as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests', 'examples',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)