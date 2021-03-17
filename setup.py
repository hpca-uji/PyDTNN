#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

from Cython.Build import cythonize
from setuptools import setup, find_packages

from settings import Settings
from setup_extra import DevelopAndPostDevelop, InstallAndPostInstall

# Common settings used by distutils and cx_freeze
s = Settings()

# Setup
setup(
    # Application details
    name=s.name,
    version=s.version,
    description=s.description,
    url=s.url,
    long_description=s.long_description,
    # Author details
    author=s.author,
    author_email=s.email,
    # Application classifiers
    classifiers=s.classifiers,
    # Application keywords
    keywords=s.keywords,
    # distutils parameters
    scripts=s.scripts,
    packages=find_packages(exclude=['build', 'dist', 'distfiles']),
    package_data=s.package_data,
    data_files=s.data_files,
    install_requires=s.install_requires,
    entry_points={
        'console_scripts': [
            'pydtnn_benchmark=pydtnn.pydtnn_benchmark:main'
        ],
    },
    cmdclass={
        'develop': DevelopAndPostDevelop,
        'install': InstallAndPostInstall,
    },
    ext_modules=cythonize(s.ext_modules, language_level=3),
)
