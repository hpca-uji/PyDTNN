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

from pydtnn.cython_modules import shrink_cython
from pydtnn.utils.best_of import BestOf


def shrink_numpy(original, vpadding, hpadding, shrunk=None):
    b, c, h, w = original.shape
    new_h, new_w = h - 2 * vpadding, w - 2 * hpadding
    if shrunk is None:
        shrunk = np.empty((b, c, new_h, new_w), dtype=original.dtype, order="C")
    shrunk[...] = original[:, :, vpadding:vpadding + new_h, hpadding:hpadding + new_w]
    return shrunk


def shrink_cython_wrapper(original, vpadding, hpadding, shrunk=None):
    if shrunk is None:
        b, c, h, w = original.shape
        new_h, new_w = h - 2 * vpadding, w - 2 * hpadding
        shrunk = np.empty((b, c, new_h, new_w), dtype=original.dtype, order="C")
    shrink_cython(original, shrunk)
    return shrunk


best_shrink = BestOf(
    name="Shrink methods",
    alternatives=[
        ("numpy", shrink_numpy),
        ("cython", shrink_cython_wrapper)
    ],
    get_problem_size=lambda *args: args[0].shape,
)
