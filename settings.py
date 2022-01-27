#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import codecs  # To use a consistent encoding when opening version.py
import os
from distutils.extension import Extension
from glob import glob

import numpy


def read(file_name):
    """Reads a file and returns its content"""
    file_name = os.path.join(os.path.dirname(__file__), file_name)
    with codecs.open(file_name, encoding='utf8') as f:
        return f.read()


def get_version():
    """Gets version from 'pydtnn/version.py'."""
    version_dict = {}
    with codecs.open('pydtnn/version.py') as fp:
        exec(fp.read(), version_dict)
    return version_dict['__version__']


def get_cython_module_names():
    """Gets cython module names from pyx files in pydtnn/cython_modules/*.pyx"""
    return [os.path.basename(x)[:-4] for x in glob("pydtnn/cython_modules/*.pyx")]


class Settings:
    """
    PyDTNN package settings
    """
    name = 'pydtnn'
    version = get_version()
    description = 'Python Distributed Training of Neural Networks'
    long_description = '\n\n'.join([read('README.rst'), read('CREDITS.rst'), read('LICENSE.rst'),
                                    read('CHANGELOG.rst')])
    url = 'https://github.com/hpca-uji/PyDTNN'
    author = 'Manuel F. Dolz Zaragozá and others'
    email = 'dolzm@uji.es'
    license = 'GPLV3+'
    scripts = []
    package_data = {
        'pydtnn': ['scripts/*.sh', 'scripts/extrae.xml', 'scripts/function-list', 'scripts/*.csv'],
    }
    data_files = []
    classifiers = [
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        # -----------------------------------------------------------
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',
        # Who the project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        # License long description
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        # Python versions supported
        'Programming Language :: Python :: 3',
    ]
    keywords = ['Deep neural networks', 'Distributed parallel training', 'Python']
    install_requires = ['numpy>=1.17.2']
    ext_modules = [
        Extension(
            module,
            ["pydtnn/cython_modules/%s.pyx" % module],
            extra_compile_args=['-fopenmp', '-O3', '-march=native', '-g0'],
            extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include()],
        ) for module in get_cython_module_names()
    ]
