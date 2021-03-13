"""
Utils definitions for Python Distributed Training of Neural Networks (PyDTNN)

PyDTNN is a light-weight library for distributed Deep Learning training and
inference that offers an initial starting point for interaction with distributed
training of (and inference with) deep neural networks. PyDTNN prioritizes
simplicity over efficiency, providing an amiable user interface which enables a
flat accessing curve. To perform the training and inference processes, PyDTNN
exploits distributed inter-process parallelism (via MPI) for clusters and
intra-process (via multi-threading) parallelism to leverage the presence of
multicore processors and GPUs at node level. For that, PyDTNN uses MPI4Py for
message-passing, BLAS calls via NumPy for multicore processors and
PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

Copyright 2021 Universitat Jaume I

This file is part of PyDTNN. PyDTNN is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

PyDTNN is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details. You
should have received a copy of the GNU General Public License along with this
program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, Sergio Barrachina, Mar Catalán, Adrián Castelló"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2021, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", "Sergio Barrachina", "Mar Catalán", "Adrián Castelló"]
__date__ = "2020/03/22"

__email__ = "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"

import ctypes
import inspect
import math
import os
from abc import ABC
from ctypes.util import find_library
from glob import glob
from importlib import import_module

import numpy as np

# import NN_model
# import scipy.linalg.blas as slb
# import scipy.stats as stats
# from scipy.signal import convolve2d

try:
    # import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    from skcuda import cublas
    from pycuda.compiler import SourceModule
    import libcudnn.libcudnn as cudnn
except ModuleNotFoundError:
    pass


def load_library(name):
    """
    Loads an external library using ctypes.CDLL.

    It searches the library using ctypes.util.find_library(). If the library is
    not found, it traverses the LD_LIBRARY_PATH until it founds it. If is not in
    any of the LD_LIBRARY_PATH paths, an ImportError exception is raised.

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


def matmul_gpu(handle, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemm = {np.float32: cublas.cublasSgemm,
                np.float64: cublas.cublasDgemm}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemm(handle, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)


def matvec_gpu(handle, transA, m, n, alpha, a, lda, b, ldb, beta, c, ldc, dtype):
    try:
        gemv = {np.float32: cublas.cublasSgemv,
                np.float64: cublas.cublasDgemv}[dtype]
    except KeyError:
        print("I cannot handle %s type!\n" % dtype.__name__)
    else:
        gemv(handle, transA, m, n, alpha, a, lda, b, ldb, beta, c, ldc)


class TensorGPU:
    def __init__(self, gpuarr, tensor_format, cudnn_dtype, tensor_type="tensor", desc=None, gpudirect=False,
                 cublas=False):
        if len(gpuarr.shape) == 2:
            self.shape = (*gpuarr.shape, 1, 1)
        else:
            self.shape = gpuarr.shape
        self.size = gpuarr.size
        self.ary = gpuarr
        if gpudirect:
            self.ptr_intp = np.intp(self.ary.base.get_device_pointer())
            self.ptr = ctypes.c_void_p(int(self.ary.base.get_device_pointer()))
        else:
            self.ptr = ctypes.c_void_p(int(gpuarr.gpudata))
        if desc:
            self.desc = desc
        if tensor_type == "tensor":
            self.desc = cudnn.cudnnCreateTensorDescriptor()
            cudnn.cudnnSetTensor4dDescriptor(self.desc, tensor_format,
                                             cudnn_dtype, *self.shape)
        elif tensor_type == "filter":
            self.desc = cudnn.cudnnCreateFilterDescriptor()
            cudnn.cudnnSetFilter4dDescriptor(self.desc, cudnn_dtype,
                                             tensor_format, *self.shape)


# Loss functions for classification CNNs

metric_format = {"categorical_accuracy": "acc: %5.2f%%",
                 "categorical_cross_entropy": "cce: %.7f",
                 "binary_cross_entropy": "bce: %.7f",
                 "categorical_hinge": "hin: %.7f",
                 "categorical_mse": "mse: %.7f",
                 "categorical_mae": "mae: %.7f",
                 "regression_mse": "mse: %.7f",
                 "regression_mae": "mae: %.7f"}


class Loss(ABC):

    def __init__(self, shape, model, eps=1e-8, enable_gpu=False, dtype=np.float32):
        self.shape = shape
        self.b, self.n = shape
        self.model = model
        self.eps = eps
        self.enable_gpu = enable_gpu
        self.dtype = dtype
        if self.enable_gpu:
            self.__init_gpu_kernel__()

    def __init_gpu_kernel__(self):
        pass


class CategoricalCrossEntropy(Loss):

    def __init_gpu_kernel__(self):
        module = SourceModule("""
        __global__ void categorical_cross_entropy(T *Y_targ, T *Y_pred, T *res,
                                                  T *dx, int b, int bs, int n, float eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b){
                int i = 0, max = 0;
                T max_value = Y_targ[idx * n];
                dx[idx * n] = Y_targ[idx * n];
                for ( i = 1; i < n; i++ ) {
                    dx[idx * n + i] = Y_targ[idx * n + i];
                    if ( Y_targ[idx * n + i] > max_value ){
                        max = i;
                        max_value = Y_targ[idx * n + i];
                    }
                }
                T pred = Y_pred[idx * n + max];
                if ( pred < eps )          pred = eps;
                else if ( pred > (1-eps) ) pred = (1-eps);
                res[idx] = logf(pred);
                dx[idx * n + max] /= -(pred * bs);
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.dtype]))

        self.categorical_cross_entropy_kern = module.get_function("categorical_cross_entropy")
        self.loss = gpuarray.empty((self.b,), self.dtype)
        dx_gpu = gpuarray.empty(self.shape, self.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)
        self.stream = self.model.stream

    def __call__(self, Y_pred, Y_targ, global_batch_size):
        if self.enable_gpu:
            threads = min(self.b, 1024)
            blocks = max(self.b, 1024) // threads + 1
            self.categorical_cross_entropy_kern(Y_targ.ary, Y_pred.ary, self.loss, self.dx.ary,
                                                np.int32(self.b), np.int32(global_batch_size),
                                                np.int32(self.n), np.float32(self.eps),
                                                grid=(blocks, 1, 1), block=(threads, 1, 1),
                                                stream=self.stream)
            loss = -gpuarray.sum(self.loss).get() / self.b
            return loss, self.dx
        else:
            Y_pred = np.clip(Y_pred, a_min=self.eps, a_max=(1 - self.eps))
            b_range = np.arange(Y_pred.shape[0])
            loss = -np.sum(np.log(Y_pred[b_range, np.argmax(Y_targ, axis=1)])) / Y_pred.shape[0]
            dx = np.copy(Y_targ)
            dx_amax = np.argmax(dx, axis=1)
            dx[b_range, dx_amax] /= (-Y_pred[b_range, dx_amax] * global_batch_size)
            return loss, dx


class BinaryCrossEntropy(Loss):

    def __init_gpu_kernel__(self):
        module = SourceModule("""
        __global__ void binary_cross_entropy(T *Y_targ, T *Y_pred, T *res,
                                             T *dx, int b, int bs, int n, T eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b){
                int i = 0, max = 0;
                T pred;
                res[idx] = 0;
                for ( i = 0; i < n; i++ ) {
                    res[idx]+= logf(fmaxf((1 - Y_targ[idx * n + i] ) -
                                               Y_pred[idx * n + i], eps));
                    pred = Y_pred[idx * n + max];
                    if ( pred < eps )          pred = eps;
                    else if ( pred > (1-eps) ) pred = (1-eps);
                    dx[idx * n + i] = (-(Y_targ[idx * n + i]  / pred) +
                                   ((1 - Y_targ[idx * n + i]) / pred) ) / bs;
                }
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.dtype]))

        self.binary_cross_entropy_kern = module.get_function("binary_cross_entropy")
        self.loss = gpuarray.empty((self.b,), self.dtype)
        dx_gpu = gpuarray.empty(self.shape, self.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)
        self.stream = self.model.stream

    def __call__(self, Y_pred, Y_targ, global_batch_size):
        assert len(Y_targ.shape) == 2
        if self.enable_gpu:
            threads = min(self.b, 1024)
            blocks = max(self.b, 1024) // threads + 1
            self.binary_cross_entropy_kern(Y_targ, Y_pred, self.loss, self.dx.ary,
                                           self.b, global_batch_size, self.n, self.eps,
                                           grid=(blocks, 1, 1), block=(threads, 1, 1),
                                           stream=self.stream)
            loss = -gpuarray.sum(self.loss) / self.b
            return loss, self.dx
        else:
            b = Y_targ.shape[0]
            loss = -np.sum(np.log(np.maximum((1 - Y_targ) - Y_pred, self.eps))) / b
            Y_pred = np.clip(Y_pred, a_min=self.eps, a_max=(1 - self.eps))
            dx = (-(Y_targ / Y_pred) + ((1 - Y_targ) / (1 - Y_pred))) / global_batch_size
            return loss, dx


class Metric(ABC):

    def __init__(self, shape, model, eps=1e-8, enable_gpu=False, dtype=np.float32):
        self.shape = shape
        self.b, self.n = shape
        self.model = model
        self.eps = eps
        self.enable_gpu = enable_gpu
        self.dtype = dtype
        if self.enable_gpu:
            self.__init_gpu_kernel__()

    def __init_gpu_kernel__(self):
        pass


class CategoricalAccuracy(Metric):

    def __init_gpu_kernel__(self):
        module = SourceModule("""
        __global__ void categorical_accuracy(T *Y_targ, T *Y_pred, T *res, int b, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b){
                int i = 0, max = 0;
                T max_value = Y_pred[idx * n];
                for ( i = 1; i < n; i++ ) {
                    if ( Y_pred[idx * n + i] > max_value ){
                        max = i;
                        max_value = Y_pred[idx * n + i];
                    }
                }
                res[idx] = Y_targ[idx * n + max];
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.dtype]))
        self.categorical_accuracy_kern = module.get_function("categorical_accuracy")
        self.cost = gpuarray.empty((self.b,), self.dtype)
        self.stream = self.model.stream

    def __call__(self, Y_pred, Y_targ):
        if self.enable_gpu:
            threads = min(self.b, 1024)
            blocks = max(self.b, 1024) // threads + 1
            self.categorical_accuracy_kern(Y_targ, Y_pred, self.cost,
                                           np.int32(self.b), np.int32(self.n),
                                           grid=(blocks, 1, 1), block=(threads, 1, 1),
                                           stream=self.stream)
            return gpuarray.sum(self.cost).get() * 100 / self.b
        else:
            b = Y_targ.shape[0]
            return np.sum(Y_targ[np.arange(b), np.argmax(Y_pred, axis=1)]) * 100 / b


class CategoricalHinge(Metric):

    def __call__(self, Y_pred, Y_targ):
        pos = np.sum(Y_targ * Y_pred, axis=-1)
        neg = np.max((1.0 - Y_targ) * Y_pred, axis=-1)
        return np.mean(np.maximum(0.0, neg - pos + 1), axis=-1)


class CategoricalMSE(Metric):

    def __call__(self, Y_pred, Y_targ):
        b = Y_targ.shape[0]
        return np.square(1 - Y_pred[np.arange(b), np.argmax(Y_targ, axis=1)]).mean()


class CategoricalMAE(Metric):

    def __call__(self, Y_pred, Y_targ):
        b = Y_targ.shape[0]
        targ = np.argmax(Y_targ, axis=1)
        return np.sum(np.absolute(1 - Y_pred[np.arange(b), np.argmax(Y_targ, axis=1)]))


class RegressionMSE(Metric):

    def __call__(self, Y_pred, Y_targ):
        return np.square(Y_targ - Y_pred).mean()


class RegressionMAE(Metric):

    def __call__(self, Y_pred, Y_targ):
        return np.sum(np.absolute(Y_targ - Y_pred))


# Compatibility aliases

categorical_cross_entropy = CategoricalCrossEntropy
binary_cross_entropy = BinaryCrossEntropy
categorical_accuracy = CategoricalAccuracy
categorical_hinge = CategoricalHinge
categorical_mse = CategoricalMSE
categorical_mae = CategoricalMAE
regression_mse = RegressionMSE
regression_mae = RegressionMAE


# Some utility functions for debugging

def printf_trace(*args):
    pass
    # print(*args)


def printf(*args):
    pass
    # print(*args)


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


# The next functions have been deprecated - use them with care!

# Only for fancy im2col/col2im indexing!
def get_indices(x_shape, kh, kw, c, h, w, s=1):
    # b, c, h, w = x_shape
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, c)
    i1 = s * np.repeat(np.arange(h), w)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = s * np.tile(np.arange(w), h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)


# Only for fancy im2col/col2im indexing!
def im2col_fancy_previous(x, kh, kw, c, h, w, s=1, idx=None):
    # Expected 'x' format (b, c, h, w)
    if not idx:
        idx = get_indices(x.shape, kh, kw, c, h, w, s)
    cols = x[:, idx[0], idx[1], idx[2]].transpose(1, 2, 0).reshape(kh * kw * c, -1)
    return cols, idx


# Only for fancy im2col/col2im indexing!
def col2im_fancy_previous(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
    b, c, h, w = x_shape
    cols_reshaped = cols.reshape(c * kh * kw, -1, b).transpose(2, 0, 1)
    x = np.zeros((b, c, h, w), dtype=cols.dtype)
    if not idx:
        idx = get_indices(x_shape, kh, kw, c, ho, wo, s)
    np.add.at(x, (slice(None), idx[0], idx[1], idx[2]), cols_reshaped)
    return x, idx


# Only for fancy im2col/col2im indexing!
def im2col_fancy(x, kh, kw, c, h, w, s=1, idx=None):
    cols, idx = im2col_fancy_previous(x, kh, kw, c, h, w, s, idx)
    return cols, None


# Only for fancy im2col/col2im indexing!
def col2im_fancy(cols, x_shape, kh, kw, ho, wo, s=1, idx=None):
    # b, c, h, w = x_shape
    cols, idx = col2im_fancy_previous(cols, x_shape, kh, kw, ho, wo, s, idx)
    return cols, idx


# Only for fancy im2col/col2im indexing!
def dilate_and_pad(input_, p=0, s=1):
    if s > 1 or p > 0:
        b, c, h, w = input_.shape
        h_, w_ = (h + ((h - 1) * (s - 1)) + 2 * p, w + ((w - 1) * (s - 1)) + 2 * p)
        res = np.zeros([b, c, h_, w_])
        res[..., p:h_ - p:s, p:w_ - p:s] = input_
        return res
    return input_

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
