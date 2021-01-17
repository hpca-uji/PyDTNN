"""
Performance tests for transposing matrices

For running the tests run:
    python perftests/perf_test_conv2d_transpose.py

To obtain a profile, run:
    python3 -m cProfile -o perf_test_conv2d_transpose.prof perftests/perf_test_conv2d_transpose.py

To graphically inspect the profile, run:
    snakeviz perf_test_conv2d_transpose.prof
"""
import ctypes
import inspect
import os
import platform
import sys
from ctypes.util import find_library
from timeit import timeit

import numpy as np
from numba import njit, prange
from prettytable import PrettyTable

if True:
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from NN_transpose_cython import transpose_2d_f2c_ji_cython, transpose_2d_f2c_ij_cython


class D:
    # def __init__(self):
    #     """Default parameters"""
    #     self.b = 1  # Batch size
    #     self.c = 1  # Channels per layer
    #     self.h = 128  # Layers height
    #     self.w = 100  # Layers width
    #     self.kn = 1  # Number of filters
    #     self.kh = 16  # Filters weights height
    #     self.kw = 10  # Filters weights width
    #     self.vpadding = 1  # Vertical padding
    #     self.hpadding = 1  # Horizontal padding
    #     self.vstride = 1  # Vertical stride
    #     self.hstride = 1  # Horizontal stride

    def __init__(self, b, c, h, w, kn, kh, kw, vpadding, hpadding, vstride, hstride):
        self.b = b  # Batch size
        self.c = c  # Channels per layer
        self.h = h  # Layers height
        self.w = w  # Layers width
        self.kn = kn  # Number of filters
        self.kh = kh  # Filters weights height
        self.kw = kw  # Filters weights width
        self.vpadding = vpadding  # Vertical padding
        self.hpadding = hpadding  # Horizontal padding
        self.vstride = vstride  # Vertical stride
        self.hstride = hstride  # Horizontal stride

    @property
    def ho(self):
        return (self.h + 2 * self.vpadding - self.kh) // self.vstride + 1

    @property
    def wo(self):
        return (self.w + 2 * self.hpadding - self.kw) // self.hstride + 1

    def __repr__(self):
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
"""


class Params:
    pass


def transpose_2d_numpy(original, transposed):
    transposed[...] = original


def transpose_2d_ravel(original, transposed):
    transposed = original.ravel(order="C")


@njit(parallel=True)
def transpose_2d_numba(original, transposed):
    n0, n1 = original.shape
    for d0 in prange(n0):
        for d1 in range(n1):
            transposed[d0, d1] = original[d0, d1]


@njit(parallel=True)
def transpose_2d_2nd_numba(original, transposed):
    n0, n1 = original.shape
    for d0 in range(n0):
        for d1 in prange(n1):
            transposed[d0, d1] = original[d0, d1]


def transpose_2d_conv_gemm(original, transposed, lib, layer):
    lib.sreshapeOut_pydtnn(ctypes.c_uint(layer.kn), ctypes.c_uint(layer.b), ctypes.c_uint(layer.wo),
                           ctypes.c_uint(layer.ho),
                           ctypes.c_void_p(original.ctypes.data), ctypes.c_void_p(transposed.ctypes.data))


def time_transpose_10(layer, shape, dtype=np.float32):
    d0, d1 = shape
    original = np.random.rand(d0, d1).astype(dtype, order="F")
    numpy_transposed = np.empty((d0, d1), dtype, order="C")
    ravel_transposed = np.empty((d0, d1), dtype, order="C")
    numba_transposed = np.empty((d0, d1), dtype, order="C")
    conv_gemm_transposed = np.empty((d0, d1), dtype, order="C")
    cython_transposed = np.empty((d0, d1), dtype, order="C")
    # Load convGemm library
    path = find_library('convGemm')
    if not path:
        for current_path in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
            if os.path.exists(os.path.join(current_path, 'libconvGemm.so')):
                path = os.path.join(current_path, 'libconvGemm.so')
                break
    if not path:
        raise ImportError("Library 'libconvGemm.so' could not be found. Please add its path to LD_LIBRARY_PATH "
                          "using 'export LD_LIBRARY_PATH=libconvGemm_path:$LD_LIBRARY_PATH' before calling this "
                          "application.")
    lib = ctypes.cdll.LoadLibrary(path)
    #
    # First run
    #
    print(f"First pass {shape} (checking outputs)", sep="", end="")
    #
    print(".", sep="", end="")
    transpose_2d_numpy(original, numpy_transposed)
    #
    print(".", sep="", end="")
    transpose_2d_ravel(original, ravel_transposed)
    # #
    # print(".", sep="", end="")
    # transpose_2d_numba(original, numba_transposed)
    # assert np.allclose(numpy_transposed, numba_transposed), "numpy and numba outer differ"
    # #
    # print(".", sep="", end="")
    # transpose_2d_2nd_numba(original, numba_transposed)
    # assert np.allclose(numpy_transposed, numba_transposed), "numpy and numba 2nd loop differ"
    #
    print(".", sep="", end="")
    transpose_2d_conv_gemm(original, conv_gemm_transposed, lib, layer)
    assert np.allclose(numpy_transposed, conv_gemm_transposed), "numpy and convGemm differ"
    #
    print(".", sep="", end="")
    transpose_2d_f2c_ji_cython(original, cython_transposed)
    assert np.allclose(numpy_transposed, cython_transposed), "numpy and cython outer differ"
    #
    print(".", sep="", end="")
    transpose_2d_f2c_ij_cython(original, cython_transposed)
    assert np.allclose(numpy_transposed, cython_transposed), "numpy and cython 2nd loop differ"
    print()
    #
    # Second run
    #
    print(f"Second pass {shape} (getting times)", sep="", end="")
    print(".", sep="", end="")
    transpose_numpy_t = timeit(lambda: transpose_2d_numpy(original, numpy_transposed),
                               number=10) / 10
    print(".", sep="", end="")
    transpose_ravel_t = timeit(lambda: transpose_2d_ravel(original, numpy_transposed),
                               number=10) / 10
    # print(".", sep="", end="")
    # transposed_numba_outer_t = timeit(lambda: transpose_2d_numba(original, numpy_transposed),
    #                                   number=10) / 10
    # print(".", sep="", end="")
    # transpose_numba_2nd_t = timeit(lambda: transpose_2d_2nd_numba(original, numpy_transposed),
    #                                number=10) / 10
    print(".", sep="", end="")
    transpose_conv_gemm_t = timeit(lambda: transpose_2d_conv_gemm(original, conv_gemm_transposed, lib, layer),
                                   number=10) / 10
    print(".", sep="", end="")
    transpose_cython_ji_t = timeit(lambda: transpose_2d_f2c_ji_cython(original, cython_transposed),
                                   number=10) / 10
    print(".", sep="", end="")
    transpose_cython_ij_t = timeit(lambda: transpose_2d_f2c_ij_cython(original, cython_transposed),
                                   number=10) / 10
    print()
    min_t = np.min([transpose_numpy_t, transpose_conv_gemm_t, transpose_cython_ji_t, transpose_cython_ij_t])
    a = "*" if transpose_conv_gemm_t == min_t else ""
    b = "*" if transpose_cython_ji_t == min_t else ""
    c = "*" if transpose_cython_ij_t == min_t else ""
    return [
        # ["numpy", "numba outer", "numba 2nd", "convGemm", "cython ji", "cython ij"],
        ["numpy", "ravel", "a", "convGemm", "b", "cython ji", "c", "cython ij"],
        ["{:6.4f}".format(transpose_numpy_t),
         "{:6.4f}".format(transpose_ravel_t),
         # transposed_numba_outer_t - transpose_numpy_t,
         # transpose_numba_2nd_t - transpose_numpy_t,
         a,
         "{:6.4f}".format(transpose_conv_gemm_t - transpose_numpy_t),
         b,
         "{:6.4f}".format(transpose_cython_ji_t - transpose_numpy_t),
         c,
         "{:6.4f}".format(transpose_cython_ij_t - transpose_numpy_t),
         ]]


if __name__ == '__main__':
    # id;height;width;channels;kernel_height;kernel_width;kernel_num;stride;padding
    # 2;227;227;3;11;11;96;4;0
    # 4;27;27;96;5;5;256;1;0
    # 6;13;13;256;3;3;384;1;0
    # 7;13;13;384;3;3;384;1;0
    # 8;13;13;384;3;3;256;1;0
    # D(b, c, h, w, kn, kh, kw, vpadding, hpadding, vstride, hstride):
    layers = [
        # AlexNet Cifar
        D(64, 3, 32, 32, 64, 3, 3, 1, 1, 2, 2),
        D(64, 64, 8, 8, 192, 3, 3, 1, 1, 1, 1),
        D(64, 192, 4, 4, 384, 3, 3, 1, 1, 1, 1),
        D(64, 384, 4, 4, 256, 3, 3, 1, 1, 1, 1),
        D(64, 256, 4, 4, 256, 3, 3, 1, 1, 1, 1),
        # AlexNet ImageNet
        D(64, 3, 227, 227, 96, 11, 11, 1, 1, 4, 4),
        D(64, 96, 27, 27, 256, 5, 5, 1, 1, 1, 1),
        D(64, 256, 13, 13, 384, 3, 3, 1, 1, 1, 1),
        D(64, 384, 13, 13, 384, 3, 3, 1, 1, 1, 1),
        D(64, 384, 13, 13, 256, 3, 3, 1, 1, 1, 1),

    ]

    t = None
    for layer in layers:
        shape = (layer.kn, layer.b * layer.ho * layer.wo)
        headers, values = time_transpose_10(layer, shape)
        if t is None:
            t = PrettyTable(['shape', ] + headers)
        # t.set_style(PLAIN_COLUMNS)
        t.align = "r"
        t.add_row([", ".join([str(x) for x in shape]), ] + values)
    print("*************************************************")
    print("** {}  OMP_NUM_THREADS: {}".format(platform.node(), os.environ["OMP_NUM_THREADS"]))
    print("** All times, except numpy, compared to numpy.")
    print(t)
