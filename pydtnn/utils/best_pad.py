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

from pydtnn.cython_modules import pad_cython
from pydtnn.utils.best_of import BestOf


def pad_numpy(matrix_in, vpadding, hpadding):
    matrix_out = np.pad(matrix_in,
                        ((0, 0), (0, 0),
                         (vpadding, vpadding), (hpadding, hpadding)),
                        mode='constant')
    return matrix_out


def pad_cython_wrapper(matrix_in, vpadding, hpadding):
    b, c, h, w = matrix_in.shape
    dtype = matrix_in.dtype
    new_h = h + 2 * vpadding
    new_w = w + 2 * hpadding
    cython_matrix_out = np.empty((b, c, new_h, new_w), dtype=dtype, order="C")
    pad_cython(matrix_in, cython_matrix_out)
    return cython_matrix_out


best_pad = BestOf(
    name="Padding methods",
    alternatives=[("numpy", pad_numpy),
                  ("cython", pad_cython_wrapper)],
    get_problem_size=lambda *args: args[0].shape,
)
