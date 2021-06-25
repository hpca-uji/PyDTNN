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


def reindex_numpy(original, h_new_indexes, v_new_indexes, indexed=None):
    if indexed is None:
        indexed = original
    if h_new_indexes is not None:
        indexed = indexed[:, :, h_new_indexes, :]
    if v_new_indexes is not None:
        indexed = indexed[:, :, :, v_new_indexes]
    if h_new_indexes is not None or v_new_indexes is not None:
        # @warning: The copy() is required to ensure the correct order of the underlying data of
        #           indexed. Otherwise using self.cg_x_indexed.ravel(order="K") will lead to
        #           unexpected results
        indexed = indexed.copy()
    return indexed


def reindex_cython_wrapper(original, h_new_indexes, v_new_indexes, indexed=None):
    if indexed is None:
        b, c, h, w = original.shape
        new_h = len(h_new_indexes) if h_new_indexes is not None else h
        new_w = len(v_new_indexes) if v_new_indexes is not None else w
        indexed = np.empty((b, c, new_h, new_w), dtype=original.dtype, order="C")
    reindex_cython(h_new_indexes, v_new_indexes, original, indexed)
    return indexed


best_reindex = BestOf(
    name="Reindex methods",
    alternatives=[
        ("numpy", reindex_numpy),
        ("cython", reindex_cython_wrapper)
    ],
    get_problem_size=lambda *args: args[0].shape,
)
