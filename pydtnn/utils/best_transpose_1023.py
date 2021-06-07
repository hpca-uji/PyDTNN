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

from pydtnn.cython_modules import transpose_1023_jik_cython, transpose_1023_ijk_cython
from pydtnn.utils.best_of import BestOf


def transpose_1023_numpy(original):
    d0, d1, d2, d3 = original.shape
    transposed = np.empty((d1, d0, d2, d3), original.dtype, order="C")
    transposed[...] = original.transpose((1, 0, 2, 3))
    return transposed


def transpose_1023_ijk_cython_wrapper(original):
    d0, d1, d2, d3 = original.shape
    transposed = np.empty((d1, d0, d2, d3), original.dtype, order="C")
    transpose_1023_ijk_cython(original, transposed)
    return transposed


def transpose_1023_jik_cython_wrapper(original):
    d0, d1, d2, d3 = original.shape
    transposed = np.empty((d1, d0, d2, d3), original.dtype, order="C")
    transpose_1023_jik_cython(original, transposed)
    return transposed


best_transpose_1023 = BestOf(
    name="Transpose 1023 methods",
    alternatives=[("numpy", transpose_1023_numpy),
                  ("ijk_cyt", transpose_1023_ijk_cython_wrapper),
                  ("jik_cyt", transpose_1023_jik_cython_wrapper),
                  ],
    get_problem_size=lambda m: m.shape,
)
