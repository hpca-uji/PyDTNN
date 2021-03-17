"""
PyDTNN Utilities
"""

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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
# 

import ctypes
import inspect
import math
import os
from ctypes.util import find_library
from glob import glob
from importlib import import_module

import numpy as np

try:
    # noinspection PyUnresolvedReferences
    from skcuda import cublas
except (ImportError, ModuleNotFoundError):
    pass


def load_library(name):
    """
    Loads an external library using ctypes.CDLL.

    It searches the library using ctypes.util.find_library(). If the library is
    not found, it traverses the LD_LIBRARY_PATH until it finds it. If it is not
    in any of the LD_LIBRARY_PATH paths, an ImportError exception is raised.

    Parameters
    ----------
    name : str
        The library name without any prefix like lib, suffix like .so, .dylib or
        version number (this is the form used for the posix linker option -l).

    Returns
    -------
    The loaded library.
    """
    path = find_library(name)
    if path is None:
        full_name = f"lib{name}.so"
        for current_path in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
            if os.path.exists(os.path.join(current_path, full_name)):
                path = os.path.join(current_path, full_name)
                break
        else:
            # Didn't find the library
            raise ImportError(f"Library '{full_name}' could not be found. Please add its path to LD_LIBRARY_PATH "
                              f"using 'export LD_LIBRARY_PATH={name.upper()}_LIB_PATH:$LD_LIBRARY_PATH' and "
                              f"then call this application again.")
    return ctypes.CDLL(path)


def blis():
    if not hasattr(blis, "lib"):
        blis.lib = load_library("blis")
    return blis.lib


def mkl():
    if not hasattr(mkl, "lib"):
        mkl.lib = load_library("mkl_rt")
    return mkl.lib


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %sytes" % (s, size_name[i])


def get_derived_classes(base_class, module_locals):
    """
    Searches on the python files of a module for classes that are derived from
    the given base_class and automatically exposes them modifying the provided
    module_locals.

    It should be called from the __init__.py file of a module as:

        get_derived_classes(BaseClass, locals())

    Parameters
    ----------
    base_class : class
        The base class to be tested for.
    module_locals: dict
        The locals() dictionary of the caller module.
    Returns
    -------
    Nothing. Modifies the provided module_locals.
    """

    file_name = inspect.stack()[1].filename
    if file_name[-11:] != "__init__.py":
        print("Warning: the 'get_derived_classes()' function should be called from an '__init__.py' file.")
    dir_path = os.path.dirname(os.path.realpath(file_name))
    for python_file in glob(os.path.join(dir_path, '*.py')):
        directory, base_file_name = os.path.split(python_file)
        module_name = os.path.split(directory)[-1]
        if base_file_name == "__init__.py":
            continue
        module = import_module(f"pydtnn.{module_name}.{base_file_name[:-3]}")
        for attribute_name in [a_n for a_n in dir(module) if a_n not in module_locals]:
            attribute = getattr(module, attribute_name)
            if inspect.isclass(attribute):
                if issubclass(attribute, base_class):
                    module_locals[attribute_name] = attribute


# Matmul operation
# Warning: the output matrix can not be cached, as it will persist outside this method
def matmul(a, b, c=None):
    # if a.dtype == np.float32:
    #    c = slb.sgemm(1.0, a, b)
    # elif a.dtype == np.float64:
    #    c = slb.dgemm(1.0, a, b)
    # else:
    # Native numpy matmul gets more performance than scipy blas!
    if c is None:
        return a @ b
    else:
        return np.matmul(a, b, c)


def _matmul_xgemm(called_from, lib, a, b, c=None):
    order = 101  # 101 for row-major, 102 for column major data structures
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if c is None:
        c = np.ones((m, n), a.dtype, order="C")
    # trans_{a,b} = 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
    if a.flags["C_CONTIGUOUS"]:
        trans_a = 111
        lda = k
    elif a.flags["F_CONTIGUOUS"]:
        trans_a = 112
        lda = m
    else:
        raise ValueError(f"Matrix a data layout not supported by {called_from}().")
    if b.flags["C_CONTIGUOUS"]:
        trans_b = 111
        ldb = n
    elif b.flags["F_CONTIGUOUS"]:
        trans_b = 112
        ldb = k
    else:
        raise ValueError(f"Matrix b data layout not supported by {called_from}().")
    ldc = n
    alpha = 1.0
    beta = 0.0
    if a.dtype == np.float32:
        lib.cblas_sgemm(ctypes.c_int(order), ctypes.c_int(trans_a), ctypes.c_int(trans_b),
                        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k), ctypes.c_float(alpha),
                        ctypes.c_void_p(a.ctypes.data), ctypes.c_int(lda),
                        ctypes.c_void_p(b.ctypes.data), ctypes.c_int(ldb),
                        ctypes.c_float(beta), ctypes.c_void_p(c.ctypes.data), ctypes.c_int(ldc))
    elif a.dtype == np.float64:
        lib.cblas_dgemm(ctypes.c_int(order), ctypes.c_int(trans_a), ctypes.c_int(trans_b),
                        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k), ctypes.c_double(alpha),
                        ctypes.c_void_p(a.ctypes.data), ctypes.c_int(lda),
                        ctypes.c_void_p(b.ctypes.data), ctypes.c_int(ldb),
                        ctypes.c_double(beta), ctypes.c_void_p(c.ctypes.data), ctypes.c_int(ldc))
    else:
        raise ValueError(f"Type '{a.dtype}' not supported by {called_from}().")
    return c


def matmul_mkl(a, b, c=None):
    # os.environ['GOMP_CPU_AFFINITY'] = ""
    # os.environ['OMP_PLACES'] = ""
    return _matmul_xgemm("matmul_mkl", mkl(), a, b, c)


def matmul_blis(a, b, c=None):
    return _matmul_xgemm("matmul_blis", blis(), a, b, c)


def matmul_gpu(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemm = {np.float32: cublas.cublasSgemm,
                np.float64: cublas.cublasDgemm}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemm(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)


def matvec_gpu(handle, trans_a, m, n, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemv = {np.float32: cublas.cublasSgemv,
                np.float64: cublas.cublasDgemv}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemv(handle, trans_a, m, n, alpha, a, lda, b, ldb, beta, c, ldc)

###############################################################
# The next functions have been deprecated - use them with care!
###############################################################

# # Some utility functions for debugging
#
# def printf_trace(*args):
#     pass
#     # print(*args)
#
#
# def printf(*args):
#     pass
#     # print(*args)
#
# # Only for fancy im2col/col2im indexing!
# def get_indices(x_shape, kh, kw, c, h, w, s=1):
#     # b, c, h, w = x_shape
#     i0 = np.repeat(np.arange(kh), kw)
#     i0 = np.tile(i0, c)
#     i1 = s * np.repeat(np.arange(h), w)
#     j0 = np.tile(np.arange(kw), kh * c)
#     j1 = s * np.tile(np.arange(w), h)
#     i = i0.reshape(-1, 1) + i1.reshape(1, -1)
#     j = j0.reshape(-1, 1) + j1.reshape(1, -1)
#     k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
#     return k.astype(int), i.astype(int), j.astype(int)
#
#
# # Only for fancy im2col/col2im indexing!
# def im2col_fancy_previous(x, kh, kw, c, h, w, s=1, idx=None):
#     # Expected 'x' format (b, c, h, w)
#     if not idx:
#         idx = get_indices(x.shape, kh, kw, c, h, w, s)
#     cols = x[:, idx[0], idx[1], idx[2]].transpose(1, 2, 0).reshape(kh * kw * c, -1)
#     return cols, idx
#
#
# # Only for fancy im2col/col2im indexing!
# def col2im_fancy_previous(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
#     b, c, h, w = x_shape
#     cols_reshaped = cols.reshape(c * kh * kw, -1, b).transpose(2, 0, 1)
#     x = np.zeros((b, c, h, w), dtype=cols.dtype)
#     if not idx:
#         idx = get_indices(x_shape, kh, kw, c, ho, wo, s)
#     np.add.at(x, (slice(None), idx[0], idx[1], idx[2]), cols_reshaped)
#     return x, idx
#
#
# # Only for fancy im2col/col2im indexing!
# def im2col_fancy(x, kh, kw, c, h, w, s=1, idx=None):
#     cols, idx = im2col_fancy_previous(x, kh, kw, c, h, w, s, idx)
#     return cols, None
#
#
# # Only for fancy im2col/col2im indexing!
# def col2im_fancy(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
#     # b, c, h, w = x_shape
#     cols, idx = col2im_fancy_previous(cols, x_shape, kh, kw, ho, wo, s, idx)
#     return cols, idx
#
#
# # Only for fancy im2col/col2im indexing!
# def dilate_and_pad(input_, p=0, s=1):
#     if s > 1 or p > 0:
#         b, c, h, w = input_.shape
#         h_, w_ = (h + ((h - 1) * (s - 1)) + 2 * p, w + ((w - 1) * (s - 1)) + 2 * p)
#         res = np.zeros([b, c, h_, w_])
#         res[..., p:h_ - p:s, p:w_ - p:s] = input_
#         return res
#     return input_

# @warning: im2col_indices is undefined
# # Only for fancy im2col/col2im indexing!
# def convolve(input_, weights, biases, p=0, s=1):
#     h, w, ci, b = input_.shape
#     co, kh, kw, ci = weights.shape
#     ho = int((h + 2 * p - kh) / s + 1)
#     wo = int((w + 2 * p - kw) / s + 1)
#     input_ = input_.transpose(3, 2, 0, 1)  # b, c, h, w, this is needed for padding
#     patched_matrix = im2col_indices(input_, kh, kw, ci, ho, wo, p, s)
#     patched_weights = weights.transpose(0, 3, 1, 2).reshape(co, -1)
#     out = ((patched_weights @ patched_matrix).T + biases).T
#     out = out.reshape(co, ho, wo, b)
#     out = out.transpose(1, 2, 0, 3)  # PyNN format
#     return out


# @warning: p and s are undefined
# # Only for fancy im2col/col2im indexing!
# def convolve_scipy(input_, weights, biases):
#     """ Does not support padding nor stride!! """
#     h, w, ci, b = input_.shape
#     co, kh, kw, ci = weights.shape
#     ho = int((h + 2 * p - kh) / s + 1)
#     wo = int((w + 2 * p - kw) / s + 1)
#     z = np.zeros([ho, wo, co, b])
#     for b_ in range(b):
#         for co_ in range(co):
#             for ci_ in range(ci):
#                 z[..., co_, b_] += convolve2d(input_[..., ci_, b_], np.rot90(weights[co_, ..., ci_], 2), mode='valid')
#             z[..., co_, b_] += biases[co_]
#     return z
