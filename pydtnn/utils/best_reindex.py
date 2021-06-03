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

from pydtnn.cython_modules import reindex_cython
from pydtnn.utils.best_of import BestOf


def reindex_numpy(h_new_indexes, v_new_indexes, matrix_in):
    matrix_out = matrix_in
    if h_new_indexes is not None:
        matrix_out = matrix_out[:, :, h_new_indexes, :]
    if v_new_indexes is not None:
        matrix_out = matrix_out[:, :, :, v_new_indexes]
    if h_new_indexes is not None or v_new_indexes is not None:
        # @warning: The copy() is required to ensure the correct order of the underlying data of
        #           matrix_out. Otherwise using self.cg_x_indexed.ravel(order="K") will lead to
        #           unexpected results
        matrix_out = matrix_out.copy()
    return matrix_out


def reindex_cython_wrapper(h_new_indexes, v_new_indexes, matrix_in):
    b, c, h, w = matrix_in.shape
    new_h = len(h_new_indexes) if h_new_indexes is not None else h
    new_w = len(v_new_indexes) if v_new_indexes is not None else w
    cython_matrix_out = np.empty((b, c, new_h, new_w), dtype=matrix_in.dtype, order="C")
    reindex_cython(h_new_indexes, v_new_indexes, matrix_in, cython_matrix_out)
    return cython_matrix_out


best_reindex = BestOf(
    name="Reindex methods",
    alternatives=[("numpy", reindex_numpy),
                  ("cython", reindex_cython_wrapper)],
    get_problem_size=lambda *args: args[2].shape,
)
