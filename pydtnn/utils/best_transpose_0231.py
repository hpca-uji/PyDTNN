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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from pydtnn.cython_modules import transpose_0231_ijk_cython, transpose_0231_ikj_cython
from pydtnn.utils.best_of import BestOf


def transpose_0231_numpy(original, transposed=None):
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d2, d3, d1), original.dtype, order="C")
    transposed[...] = original.transpose((0, 2, 3, 1))
    return transposed


def transpose_0231_ijk_cython_wrapper(original, transposed=None):
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d2, d3, d1), original.dtype, order="C")
    transpose_0231_ijk_cython(original, transposed)
    return transposed


def transpose_0231_ikj_cython_wrapper(original, transposed=None):
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d0, d2, d3, d1), original.dtype, order="C")
    transpose_0231_ikj_cython(original, transposed)
    return transposed


best_transpose_0231 = BestOf(
    name="Transpose 0231 methods",
    alternatives=[
        ("ikj_cyt", transpose_0231_ikj_cython_wrapper),
        ("ijk_cyt", transpose_0231_ijk_cython_wrapper),
        ("numpy", transpose_0231_numpy),
    ],
    get_problem_size=lambda *args: args[0].shape,
)
