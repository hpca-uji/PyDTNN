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

from pydtnn.cython_modules import transpose_2d_f2c_ji_cython, transpose_2d_f2c_ij_cython
from pydtnn.utils import load_library
from pydtnn.utils.best_of import BestOf

cg_lib = load_library("convGemm")


def transpose_2d_numpy(original):
    d0, d1 = original.shape
    transposed = np.empty((d0, d1), original.dtype, order="C")
    transposed[...] = original
    return transposed


def transpose_2d_ravel(original):
    d0, d1 = original.shape
    transposed = original.ravel(order="C").reshape(d0, d1)
    return transposed


def transpose_2d_conv_gemm(original):
    d0, d1 = original.shape
    transposed = np.empty((d0, d1), original.dtype, order="C")
    cg_lib.sreshapeOut_pydtnn(ctypes.c_uint(d0), ctypes.c_uint(d1), ctypes.c_uint(1),
                              ctypes.c_uint(1),
                              ctypes.c_void_p(original.ctypes.data), ctypes.c_void_p(transposed.ctypes.data))
    return transposed


def transpose_2d_f2c_ji_cython_wrapper(original):
    d0, d1 = original.shape
    transposed = np.empty((d0, d1), original.dtype, order="C")
    transpose_2d_f2c_ji_cython(original, transposed)
    return transposed


def transpose_2d_f2c_ij_cython_wrapper(original):
    d0, d1 = original.shape
    transposed = np.empty((d0, d1), original.dtype, order="C")
    transpose_2d_f2c_ij_cython(original, transposed)
    return transposed


best_transpose_2d_f2c = BestOf(
    name="Transpose 2D f2c methods",
    alternatives=[("numpy", transpose_2d_numpy),
                  # ("ravel", transpose_2d_ravel), # Same time as the numpy variant
                  ("convGemm", transpose_2d_conv_gemm),
                  ("ji_cyt", transpose_2d_f2c_ji_cython_wrapper),
                  ("ij_cyt", transpose_2d_f2c_ij_cython_wrapper),
                  ],
    get_problem_size=lambda m: m.shape,
)

#
# Numba legacy code
#
# from numba import njit, prange
#
# @njit(parallel=True)
# def transpose_2d_numba(original, transposed):
#     n0, n1 = original.shape
#     for d0 in prange(n0):
#         for d1 in range(n1):
#             transposed[d0, d1] = original[d0, d1]
#
#
# @njit(parallel=True)
# def transpose_2d_2nd_numba(original, transposed):
#     n0, n1 = original.shape
#     for d0 in range(n0):
#         for d1 in prange(n1):
#             transposed[d0, d1] = original[d0, d1]
