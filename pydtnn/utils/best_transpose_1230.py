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

import ctypes

import numpy as np

from pydtnn.cython_modules import transpose_1230_ij_cython, transpose_1230_ji_cython
from pydtnn.utils import load_library
from pydtnn.utils.best_of import BestOf

cg_lib = load_library("convGemm")


def transpose_1230_numpy(original, transposed=None):
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d1, d2, d3, d0), original.dtype, order="C")
    transposed[...] = original.transpose((1, 2, 3, 0))
    return transposed


def transpose_1230_conv_gemm(original, transposed=None):
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d1, d2, d3, d0), original.dtype, order="C")
    cg_lib.sreshapeWeights_pydtnn(ctypes.c_uint(d0), ctypes.c_uint(d1), ctypes.c_uint(d3), ctypes.c_uint(d2),
                                  ctypes.c_void_p(original.ctypes.data), ctypes.c_void_p(transposed.ctypes.data))
    return transposed


def transpose_1230_ij_cython_wrapper(original, transposed=None):
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d1, d2, d3, d0), original.dtype, order="C")
    transpose_1230_ij_cython(original, transposed)
    return transposed


def transpose_1230_ji_cython_wrapper(original, transposed=None):
    d0, d1, d2, d3 = original.shape
    if transposed is None:
        transposed = np.empty((d1, d2, d3, d0), original.dtype, order="C")
    transpose_1230_ji_cython(original, transposed)
    return transposed


best_transpose_1230 = BestOf(
    name="Transpose 1230 methods",
    alternatives=[
        ("ji_cyt", transpose_1230_ji_cython_wrapper),
        ("ij_cyt", transpose_1230_ij_cython_wrapper),
        ("convGemm", transpose_1230_conv_gemm),
        ("numpy", transpose_1230_numpy),
    ],
    get_problem_size=lambda *args: args[0].shape,
)
