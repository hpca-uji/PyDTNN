#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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

from pathlib import Path

from setuptools import setup, find_packages, Extension

import numpy
from Cython.Build import cythonize


setup(
    packages=find_packages(include=["pydtnn", "pydtnn.*"]),
    ext_modules=cythonize([
        Extension(
            ".".join(pyx.with_suffix("").parts),
            [str(pyx)],
            extra_compile_args=["-fopenmp", "-O3", "-march=native", "-g0"],
            extra_link_args=["-fopenmp"],
            include_dirs=[numpy.get_include()],
        ) for pyx in Path("pydtnn").rglob("*.pyx")
    ], language_level=3),
)
