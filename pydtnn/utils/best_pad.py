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


def pad_assign(original, vpadding, hpadding, padded=None):
    b, c, h, w = original.shape
    new_h, new_w = h + 2 * vpadding, w + 2 * hpadding
    if padded is None:
        padded = np.zeros((b, c, new_h, new_w), original.dtype)
    padded[:, :, vpadding:new_h - vpadding, hpadding:new_w - hpadding] = original
    return padded


def pad_numpy(original, vpadding, hpadding, padded=None):
    if padded is None:
        padded = np.pad(original,
                        ((0, 0), (0, 0),
                         (vpadding, vpadding), (hpadding, hpadding)),
                        mode='constant')
    else:
        padded[...] = np.pad(original,
                             ((0, 0), (0, 0),
                              (vpadding, vpadding), (hpadding, hpadding)),
                             mode='constant')
    return padded


def pad_cython_wrapper(original, vpadding, hpadding, padded=None):
    if padded is None:
        b, c, h, w = original.shape
        new_h, new_w = h + 2 * vpadding, w + 2 * hpadding
        padded = np.empty((b, c, new_h, new_w), dtype=original.dtype, order="C")
    pad_cython(original, padded)
    return padded


best_pad = BestOf(
    name="Padding methods",
    alternatives=[
        ("assign", pad_assign),
        ("numpy", pad_numpy),
        ("cython", pad_cython_wrapper),
    ],
    get_problem_size=lambda *args: args[0].shape,
)
