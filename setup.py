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
    url='https://github.com/ShuaibinLi/ESBox',
    packages=_find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    install_requires=[
        'pyyaml',
        'tensorboardX',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    entry_points={"console_scripts": ["esbox=esbox.example:main"]},
)
