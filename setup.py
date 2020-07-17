#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup

with open(os.path.join('networkheatdiffusion', '__init__.py')) as ver_file:
    for line in ver_file:
        if line.startswith('__version__'):
            version=re.sub("'", "", line[line.index("'"):])

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'ndex2',
    'networkx',
    'requests',
    'scipy',
    'numpy'
]

test_requirements = [
    'requests-mock',
]

setup(
    name='networkheatdiffusion',
    version=version,
    description="Package to run Heat Diffusion locally or via service",
    long_description=readme + '\n\n' + history,
    author="Chris Churas",
    author_email='churas.camera@gmail.com',
    url='https://github.com/idekerlab/networkheatdiffusion',
    packages=[
        'networkheatdiffusion',
    ],
    package_dir={'networkheatdiffusion':
                 'networkheatdiffusion'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='networkheatdiffusion',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
