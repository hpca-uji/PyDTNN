""" Python Distributed Training of Neural Networks - PyDTNN

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

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details. You
should have received a copy of the GNU General Public License along with this
program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, Sergio Barrachina, Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", "Sergio Barrachina", "Mar Catalan", "Adrian Castello"]
__date__ = "2020/11/14"

__email__ = "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Testing"
__version__ = "1.1.0"

import os

import numpy as np
from math import floor

import ctypes
from ctypes.util import find_library


# Credits of this function go to 'rotglug' stackoverflow user
# https://stackoverflow.com/questions/8658813/control-memory-alignment-in-python-ctypes
# Some changes have been done to the original code
def ctypes_aligned_alloc(alignment, size):
    buf_size = size + (alignment - 1)
    raw_memory = bytearray(buf_size)
    ctypes_raw_type = (ctypes.c_byte * buf_size)
    ctypes_raw_memory = ctypes_raw_type.from_buffer(raw_memory)
    raw_address = ctypes.addressof(ctypes_raw_memory)
    offset = raw_address % alignment
    offset_to_aligned = alignment - offset
    ctypes_aligned_type = (ctypes.c_byte * size)
    ctypes_aligned_memory = ctypes_aligned_type.from_buffer(raw_memory, offset_to_aligned)
    return ctypes_aligned_memory


class GemmConv:
    """Exposes the libgemmConv functions following the PyDTNN conventions"""

    # gemmConv.h
    # ----------
    # #define BLOCK_MC 560  // 120
    # #define BLOCK_KC 368  // 640
    # #define BLOCK_NC 3072
    # convEval.c
    # ----------
    # Ac_pack = (float*) aligned_alloc(4096,BLOCK_MC*BLOCK_KC*sizeof(float));
    # Bc_pack = (float*) aligned_alloc(4096,BLOCK_KC*BLOCK_NC*sizeof(float));

    def __init__(self, alignment=4096, block_mc=560, block_kc=368, block_nc=3072, dtype=np.float32):
        """
        Loads the libgemmConv.so library and creates the required auxiliary matrices ac_pack and bc_pack
        """
        path = find_library('gemmConv')
        if not path:
            for current_path in os.environ['LD_LIBRARY_PATH'].split(':'):
                if os.path.exists(os.path.join(current_path, 'libgemmConv.so')):
                    path = os.path.join(current_path, 'libgemmConv.so')
                    break
        if not path:
            raise ImportError("Library 'libgemmConv.so' could not be found. Please add its path to LD_LIBRARY_PATH "
                              "using 'export LD_LIBRARY_PATH=libgemmConv_path:$LD_LIBRARY_PATH' before calling this "
                              "application.")
        self.lib = ctypes.cdll.LoadLibrary(path)
        self.dtype = dtype
        dtype_bytes = dtype(1).nbytes
        self.ac_pack = ctypes_aligned_alloc(alignment, block_mc * block_kc * dtype_bytes)
        self.bc_pack = ctypes_aligned_alloc(alignment, block_kc * block_nc * dtype_bytes)

    def gemm_conv(self, filters, layers, biases=None, alpha=1.0, beta=1.0,
                  vpadding=0, hpadding=0, vstride=1, hstride=1):
        """
        Calls a libgemmConv function to perform a matrix matrix multiplication with an implicit im2col.

        The matrix matrix product is in the form C = alpha * AB + beta * C, where:
            + A is the filters matrix
            + B is the im2col(layers) matrix
            + C is the biases matrix
        """
        if vpadding != hpadding:
            raise ValueError("gemmConv does not support different vertical and horizontal paddings")
        if vstride != hstride:
            raise ValueError("gemmConv does not support different vertical and horizontal strides")
        stride = vstride

        kn, ck, kh, kw = filters.shape
        b, c, h, w = layers.shape
        assert ck == c, "Number of channels in filters and layers should be the same!"

        ho = floor((h + 2 * vpadding - kh) / vstride) + 1
        wo = floor((w + 2 * hpadding - kw) / hstride) + 1

        # offset matrix
        if biases is None:
            biases = np.zeros((kn, b * ho * wo)).astype(filters.dtype, order='F')

        assert filters.dtype == layers.dtype == biases.dtype, \
            "All the matrices must have the same type of data!"

        assert filters.dtype == self.dtype, \
            "The matrices must have the same type of data as the one specified when " \
            "the class was instantiated!"

        # gemmConv.c
        # ----------
        # * @param[in] kh Kernel height.
        # * @param[in] kw Kernel width.
        # * @param[in] c Number of channels of input tensor.
        # * @param[in] kn Kernel number.
        # * @param[in] alpha Scalar alpha.
        # * @param[in] A Matrix A. lda assumed as kn.
        # * @param[in] h Input tensor height.
        # * @param[in] w Input tensor width.
        # * @param[in] b Batch size.
        # * @param[in] stride Stride to apply the kernels to the input tensor.
        # * @param[in] In 1D-array containing a flattened version of the input tensor.
        # * @param[in] beta Scalar beta.
        # * @param[in,out] C Matrix C. ldc assumed as kn.
        # * @param[in] Ac_pack Workspace for the packing of A (Only ofr allocation purposes).
        # * @param[in] Bc_pack Workspace for the packing of B (Only ofr allocation purposes).
        # void sgemm_conv(unsigned int kh, unsigned int kw,
        #                 unsigned int c, unsigned int kn,
        #                 float alpha, float * A,
        #                 unsigned int h, unsigned int w,
        #                 unsigned int b, unsigned int stride,
        #                 float * In, float beta,
        #                 float * C,
        #                 float * Ac_pack, float * Bc_pack ){

        if filters.dtype == np.float16:
            xgemm_conv = self.lib.hgemm_conv
        elif filters.dtype == np.float32:
            xgemm_conv = self.lib.sgemm_conv
        else:
            raise ValueError("Type {} not supported by gemm_conv!".format(str(filters.dtype)))

        layers_1d = layers.flatten()

        xgemm_conv(ctypes.c_uint(kh), ctypes.c_uint(kw),
                   ctypes.c_uint(c), ctypes.c_uint(kn),
                   ctypes.c_float(alpha), ctypes.c_void_p(filters.ctypes.data),
                   ctypes.c_uint(ho), ctypes.c_uint(wo),
                   ctypes.c_uint(b), ctypes.c_uint(stride),
                   ctypes.c_void_p(layers_1d.ctypes.data), ctypes.c_float(beta),
                   ctypes.c_void_p(biases.ctypes.data),
                   ctypes.byref(self.ac_pack), ctypes.byref(self.bc_pack))
        return biases


def __usage_example__():
    # Imports for timing and testing this usage example (not required otherwise)
    from timeit import timeit
    from NN_im2col_cython import im2col_cython
    # Create filters and layers matrices
    b = 32  # Batch size
    c = 3  # Channels per layer
    h = 128  # Layers height
    w = 128  # Layers width
    kn = 16  # Number of filters
    kh = 16  # Filters height
    kw = 16  # Filters width
    padding = 0  # Padding
    stride = 1  # Stride
    layers = np.ones((b, c, h, w)).astype(np.float32, order='F')
    filters = np.zeros((kn, c, kh, kw)).astype(np.float32, order='F')
    filters[0][0][0][0] = 1.89
    filters[1][1][1][1] = 3.0
    filters[2][2][2][2] = 4.0
    print("Using gemm_conv to compute alpha * filters * layers + beta * offset...")
    gemm_conv = GemmConv()
    gemm_conv_result = gemm_conv.gemm_conv(filters, layers,
                                           vpadding=padding, hpadding=padding,
                                           vstride=stride, hstride=stride)
    print(gemm_conv_result)
    print("Sum: ", gemm_conv_result.sum())
    sgemm_conv_t = timeit(lambda: gemm_conv.gemm_conv(filters, layers,
                                                      vpadding=padding, hpadding=padding,
                                                      vstride=stride, hstride=stride),
                          number=5)

    print("gemm_conv time: {:.2f}".format(sgemm_conv_t))
    print()
    print("Using im2col and mm...")
    a_t = im2col_cython(layers, kh, kw, padding, padding, stride, stride)
    w_c = filters.reshape(kn, -1)
    im2col_mm_result = w_c @ a_t
    print(im2col_mm_result)
    print("Sum: ", im2col_mm_result.sum())
    print("np.allclose: ", np.allclose(gemm_conv_result, im2col_mm_result))
    im2col_t = timeit(lambda: im2col_cython(layers, kh, kw, padding, padding, stride, stride), number=5)
    print("im2col time: {:.2f}".format(im2col_t))
    mm_t = timeit(lambda: w_c @ a_t, number=5)
    print("mm time: {:.2f}".format(mm_t))


if __name__ == "__main__":
    __usage_example__()
