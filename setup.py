#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import sys
import os
import re
from setuptools import setup

cur_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cur_dir, 'README.md'), 'rb') as f:
    lines = [x.decode('utf-8') for x in f.readlines()]
    lines = ''.join([re.sub('^<.*>\n$', '', x) for x in lines])
    long_description = lines


def _find_packages(prefix=''):
    packages = []
    path = '.'
    prefix = prefix
    for root, _, files in os.walk(path):
        if '__init__.py' in files:
            if sys.platform == 'win32':
                packages.append(re.sub('^[^A-z0-9_]', '', root.replace('\\', '.')))
            else:
                packages.append(re.sub('^[^A-z0-9_]', '', root.replace('/', '.')))
    return packages


def read(*parts):
    with codecs.open(os.path.join(cur_dir, *parts), 'r') as fp:
        return fp.read()


# Reference: https://github.com/PaddlePaddle/PARL/blob/develop/setup.py
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


setup(
    name='esbox',
    version=find_version("esbox", "__init__.py"),
    description='Evolutionary Strategy Tools Box',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://gitlab.baidu.com/nlp-ol/ESBox',
    packages=_find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    install_requires=[
        'pyyaml',
        'parl',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={"console_scripts": ["esbox=esbox.example:main"]},
)
