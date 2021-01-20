"""
convGemm module for Python Distributed Training of Neural Networks (PyDTNN)

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

Copyright 2020 Universitat Jaume I

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

import ctypes
import os
import platform
import time
from ctypes.util import find_library

import numpy as np


class KeyDefaultDict(dict):
    def __init__(self, default_factory=None, **kwargs):
        super().__init__(self, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class ConvGemm:
    """
    Exposes the libconvGemm functions following the PyDTNN conventions.

    Methods
    -------
    conv_gemm(weights, x, biases, alpha, beta, vpadding, hpadding, vstride, hstride, biases_vector)
        Calls the appropriate convGemm function from libconvGemm.so to perform a
        matrix matrix multiplication with an implicit im2col.

    Examples
    --------
    See __usage_example__() method for an example of use. This example can be
    run with: 'python NN_conv_gemm.py'

    Tests
    -----
    To perform the tests, run the following command from the current directory:
        python -m unittest unittests.TestConvGemm

    (see unittests/test_NN_conv_gemm.py for more instructions on testing)
    """

    lib_cg = None  # will link to the libconvGemm.so library
    biases_cg_cache = None
    weights_cg_cache = None
    # out_cg_cache = None  # Warning: don't use a cached matrix for the output

    def __init__(self, dtype=np.float32, debug=False):
        """
        Loads the libconvGemm.so library and creates the required auxiliary matrices ac_pack and bc_pack.

        Parameters
        ----------
        dtype : data type
            The element data type being used on all the matrices.
        debug : boolean
            Whether to print debug information or not.
        """
        if isinstance(dtype, type):
            self.dtype = dtype
        else:
            try:
                self.dtype = {'float32': np.float32, 'float64': np.float64}[dtype]
            except KeyError:
                raise AttributeError("dtype '{}' not recognized".format(dtype)) from None
        if ConvGemm.lib_cg is None:
            path = find_library('convGemm')
            if not path:
                for current_path in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
                    if os.path.exists(os.path.join(current_path, 'libconvGemm.so')):
                        path = os.path.join(current_path, 'libconvGemm.so')
                        break
            if not path:
                raise ImportError("Library 'libconvGemm.so' could not be found. Please add its path to LD_LIBRARY_PATH "
                                  "using 'export LD_LIBRARY_PATH=libconvGemm_path:$LD_LIBRARY_PATH' before calling "
                                  "this application.")
            ConvGemm.lib_cg = ctypes.cdll.LoadLibrary(path)
            ConvGemm.biases_cg_cache = KeyDefaultDict(lambda shape: np.empty(shape, self.dtype, order="F"))
            ConvGemm.weights_cg_cache = KeyDefaultDict(lambda shape: np.empty(shape, self.dtype, order="C"))

        # Declare ac_pack and bc_pack and allocate space for them
        # @todo: The next fragment of code should be dependant on the architecture and the dtype
        self.ac_pack = ctypes.POINTER(ctypes.c_float)()
        self.bc_pack = ctypes.POINTER(ctypes.c_float)()
        self.lib_cg.alloc_pack_buffs.restype = ctypes.c_int
        result = self.lib_cg.alloc_pack_buffs(ctypes.byref(self.ac_pack), ctypes.byref(self.bc_pack))
        if result == 1:
            raise MemoryError("Could not allocate space for ac_pack or bc_pack!")
        self.debug = debug
        if not self.debug:
            time.perf_counter = lambda: 0
        # Choose the appropriate convGemm function depending on the architecture and the data type being used
        if platform.machine() == 'aarch64':
            if self.dtype == np.float16:
                self.x_conv_gemm = self.lib_cg.hconvGemm
            elif self.dtype == np.float32:
                self.x_conv_gemm = self.lib_cg.sconvGemm
            else:
                raise ValueError("Type {} not supported by this version of libconvGemm!".format(str(self.dtype)))
        elif platform.machine() == 'x86_64':
            if self.dtype == np.float32:
                self.x_conv_gemm = self.lib_cg.sconvGemm
                self.x_apply_bias = self.lib_cg.sapplyBias
            else:
                raise ValueError("Type {} not supported by this version of libconvGemm!".format(str(self.dtype)))
        else:
            raise ValueError("Platform '{}' not yet supported")

    def __del__(self):
        """Free the allocated matrices"""

        def find_msvcr():
            import re
            import sys
            exec_bytes = open(sys.executable, "rb").read()
            match = re.search("msvcr([0-9]+|t).dll", str(exec_bytes), re.IGNORECASE)
            return match.group(0)

        if platform.system() == 'Windows':
            libc = ctypes.cdll.LoadLibrary(find_msvcr())
        elif platform.system() == 'Linux':
            libc = ctypes.cdll.LoadLibrary('libc.so.6')
        elif platform.system == 'Darwin':
            libc = ctypes.cdll.LoadLibrary('libc.dylib')
        else:
            raise AssertionError("Don't know how to get to libc for a '{}' system".format(platform.system()))
        try:
            libc.free(self.ac_pack)
            libc.free(self.bc_pack)
        except AttributeError:
            pass

    def conv_gemm(self, weights, x, biases=None, alpha=1.0, beta=1.0, vpadding=0, hpadding=0, vstride=1, hstride=1,
                  biases_vector=None):
        """
        Calls the appropriate convGemm function from libconvGemm.so to perform a
        matrix matrix multiplication with an implicit im2col.

        The matrix matrix product is in the form C = alpha * A * B + beta * C, where:
            + A is the weights matrix,
            + B is the im2col(x) matrix, and
            + C is the biases matrix.

        If the biases vector is supplied, the xapplyBias function of the libconvGemm library will be called. This
        function sums each element of the biases vector to all the elements in the corresponding output channel.

        Parameters
        ----------
        weights : array_like
            The weights matrix (kn x c x kh x kw).
        x : array_like
            The layers matrix (b x c x h x w).
        biases : array_like
            An optional biases matrix (kn x b*ho*wo).
        alpha : float
            The alpha factor.
        beta : float
            The beta factor.
        vpadding : int
            The vertical padding to be applied to the x matrix.
        hpadding : int
            The horizontal padding to be applied to the x matrix.
        vstride : int
            The vertical stride.
        hstride : int
            The horizontal stride.
        biases_vector: array_like
            The biases that have to be summed to all the elements in each output channel.

        Returns
        -------
        array_like
            The result of alpha * weights * im2col(x_padded) + beta * biases.
        """

        tic01 = time.perf_counter()

        # Pad x matrix and set vpadding and hpadding to 0
        if vpadding == 0 and hpadding == 0:
            x_padded = x
        else:
            # Hand made alternative to:
            #  x_padded = np.pad(x, ((0, 0), (0, 0), (vpadding, vpadding), (hpadding, hpadding)), mode='constant')
            b, c, h, w = x.shape
            new_h , new_w = h + 2 * vpadding, w + 2 * hpadding
            x_padded = np.zeros((b, c, new_h, new_w), x.dtype)
            x_padded[:, :, vpadding:new_h-vpadding, hpadding:new_w-hpadding] = x
            vpadding = hpadding = 0

        tic02 = time.perf_counter()

        # Get matrices dimensions (once x matrix has been padded)
        kn, ck, kh, kw = weights.shape
        b, c, h, w = x_padded.shape
        assert ck == c, "Number of channels in weights and x should be the same!"

        # Compute height and weight of the output
        # Note: h and w are obtained from x_padded (no from x) and vpadding and
        #       hpadding are set to 0 in order to use the usual formulation
        ho = (h + 2 * vpadding - kh) // vstride + 1
        wo = (w + 2 * hpadding - kw) // hstride + 1

        # Create zero biases matrix if none provided. Otherwise, test its dimensions
        if biases is None:
            # @warning: np.empty() could be used instead of np.zeros() only if the underlying operation does not
            # sum the biases when beta is 0.0. Otherwise, as the non initialized biases matrix could have nan elements,
            # the output will also have nan elements.
            beta = 0.0
            # biases_cg = np.empty((kn, b * ho * wo), weights.dtype, order='F')
            biases_cg = self.biases_cg_cache[(kn, b * ho * wo)]
        else:
            # biases = biases.copy()  # To avoid overriding the original biases matrix
            biases_cg = biases.astype(weights.dtype, order="F")
            # biases_cg = np.empty((kn, b * ho * wo), weights.dtype, order='F')
            # biases_cg[...] = biases
            assert (kn, b * ho * wo) == biases.shape, \
                "Biases matrix should be ({}, {}), instead it is {}".format(kn, b * ho * wo, biases.shape)

        tic03 = time.perf_counter()

        # Check that dtype is the same on all the matrices
        assert weights.dtype == x.dtype == biases_cg.dtype, \
            "All the matrices must have the same type of data!"
        assert weights.dtype == self.dtype, \
            "The input matrices must have the same type of data as the one specified when " \
            "this class was instantiated!"

        # Change matrices axes to the convGemm expected order:
        #   Where I→hi×wi×ci×b corresponds to the input tensor, | PyDTNN (C order): b×ci×hi×wi   (F order: wi×hi×ci×b)
        #   F→kn×kh×kw×ci denotes the filters,                  | PyDTNN (C order): kn×ci×kh×kw  (F order: kw×kh×ci×kn)
        #   and O→kn×(ho·wo·b) is the output tensor             | PyDTNN (C order): kn×(b·ho·wo) (F order: (wo·ho·b)×kn)

        # PREVIOUS(hxw): weights_cg = weights.transpose((0, 2, 3, 1)).reshape((kn, -1), order="F")
        # NEW(wxh): 1) weights_cg = weights.transpose((0, 3, 2, 1)).flatten(order="F")
        # NEW(wxh): 2) weights_cg = weights.transpose((1, 2, 3, 0)).flatten(order="C")
        # NEW(wxh): 3)
        # weights_cg = weights.transpose((1, 2, 3, 0)).ravel(order="C")

        # NEW(wxh): 4) (best until sreshapeWeights_pydtnn())
        # weights_cg = np.empty((c, kh, kw, kn), weights.dtype, order="C")
        # weights_cg[...] = weights.transpose((1, 2, 3, 0))
        # weights_cg.ravel(order="K")

        # NEW(wxh): 5)
        # void sreshapeWeights_pydtnn(unsigned int kn, unsigned int c, unsigned int kh, unsigned int kw,
        #                             float* weights_pydtnn, float* restrict weights);
        # weights_cg = np.empty((c, kh, kw, kn), weights.dtype, order="C")
        weights_cg = self.weights_cg_cache[(c, kh, kw, kn)]
        self.lib_cg.sreshapeWeights_pydtnn(ctypes.c_uint(kn), ctypes.c_uint(c), ctypes.c_uint(kw), ctypes.c_uint(kh),
                                           ctypes.c_void_p(weights.ctypes.data),
                                           ctypes.c_void_p(weights_cg.ctypes.data))

        # NEW(wxh): 6)
        # weights_cg = np.empty((c, kh, kw, kn), weights.dtype, order="C")
        # transpose_1230_2nd_cython(weights, weights_cg)

        tic04 = time.perf_counter()

        # PREVIOUS(hxw): x_padded_cg = x_padded.transpose((2, 3, 1, 0)).flatten(order="F")
        # NEW(wxh) 1): x_padded_cg = x_padded.transpose((3, 2, 1, 0)).flatten(order="F")
        # NEW(wxh) 2): x_padded_cg = x_padded.ravel(order="C")
        # NEW(wxh) 3):
        x_padded_cg = x_padded.ravel(order="K")

        tic05 = time.perf_counter()

        # Call custom added function to libconvGemm.so to print the received parameters
        if self.debug:
            try:
                self.lib_cg.expose_sconvGemm(ctypes.c_uint(kw), ctypes.c_uint(kh),
                                             ctypes.c_uint(c), ctypes.c_uint(kn),
                                             ctypes.c_float(alpha), ctypes.c_void_p(weights_cg.ctypes.data),
                                             ctypes.c_uint(w), ctypes.c_uint(h),
                                             ctypes.c_uint(b), ctypes.c_uint(hstride), ctypes.c_uint(vstride),
                                             ctypes.c_void_p(x_padded_cg.ctypes.data), ctypes.c_float(beta),
                                             ctypes.c_void_p(biases_cg.ctypes.data),
                                             self.ac_pack, self.bc_pack)
            except AttributeError:
                print("Warning: Custom 'expose_sconvGemm' function is not present in 'libconvGemm.so'. "
                      "You can safely ignore this warning.")

        # Call the appropriate convGemm function from libconvGemm
        self.x_conv_gemm(ctypes.c_uint(kw), ctypes.c_uint(kh),
                         ctypes.c_uint(c), ctypes.c_uint(kn),
                         ctypes.c_float(alpha), ctypes.c_void_p(weights_cg.ctypes.data),
                         ctypes.c_uint(w), ctypes.c_uint(h),
                         ctypes.c_uint(b), ctypes.c_uint(hstride), ctypes.c_uint(vstride),
                         ctypes.c_void_p(x_padded_cg.ctypes.data), ctypes.c_float(beta),
                         ctypes.c_void_p(biases_cg.ctypes.data),
                         self.ac_pack, self.bc_pack)

        tic06 = time.perf_counter()

        if biases_vector is not None:
            self.x_apply_bias(ctypes.c_uint(kn), ctypes.c_uint(b * ho * wo),
                              ctypes.c_void_p(biases_vector.ctypes.data), ctypes.c_void_p(biases_cg.ctypes.data),
                              ctypes.c_uint(kn))

        tic07 = time.perf_counter()

        # Change output matrix to the PyDTNN expected order:
        # * PREVIOUS(hxw): out = biases_cg.reshape((kn, b, wo, ho)).transpose((0, 1, 3, 2)).reshape(kn, -1, order="C")
        # * NEW(wxh) 1): out = biases_cg.reshape((kn, b, ho, wo)).transpose((0, 1, 2, 3)).reshape(kn, -1, order="C")
        # * NEW(wxh) 2):
        #     works:  out = biases_cg.reshape((kn, b, ho, wo)).reshape((kn, -1), order="C")
        #     slower: out = np.empty((kn, b*ho*wo), biases_cg.dtype, order="C")
        #             out[...] = biases_cg
        #     best:
        #             out = biases_cg.reshape((kn, -1), order="C")
        # * NEW(wxh) 3):
        #     void sreshapeOut_pydtnn(unsigned int kn, unsigned int b, unsigned int h, unsigned int w,
        #                             float*  out, float* restrict reshaped);
        # out = self.out_cg_cache[(kn, b * ho * wo)]  # out can persist outside, don't use this
        out = np.empty((kn, b * ho * wo), weights.dtype, order="C")
        self.lib_cg.sreshapeOut_pydtnn(ctypes.c_uint(kn), ctypes.c_uint(b), ctypes.c_uint(wo), ctypes.c_uint(ho),
                                       ctypes.c_void_p(biases_cg.ctypes.data), ctypes.c_void_p(out.ctypes.data))
        # * NEW(wxh) 4):
        # out = np.empty((kn, b * ho * wo), weights.dtype, order="C")
        # transpose_2d_f2c_ji_cython(biases_cg, out)

        tic08 = time.perf_counter()

        if self.debug:
            print(f"conv_gemm:")
            print(f"  padding:            {tic02 - tic01:0.4f} s")
            print(f"  biases:             {tic03 - tic02:0.4f} s")
            print(f"  weights_cg:         {tic04 - tic03:0.4f} s     weights.shape: {weights.shape}")
            print(f"  x_padded_cg:        {tic05 - tic04:0.4f} s     x.shape: {x.shape}")
            print(f"  before x_conv_gemm: {tic05 - tic01:0.4f} s")
            print(f"  x_conv_gemm:        {tic06 - tic05:0.4f} s")
            print(f"  x_apply_bias:       {tic07 - tic06:0.4f} s")
            print(f"  after x_conv_gemm:  {tic08 - tic07:0.4f} s")
            print(f"  total:              {tic08 - tic01:0.4f} s")

        return out


def __usage_example__():
    # Imports for this usage example (not required otherwise)
    from timeit import timeit
    from NN_im2col_cython import im2col_cython
    # Default parameters (1st layer AlexNet for Cifar10)
    b = 64  # Batch size
    c = 3  # Channels per layer
    h = 32  # Layers height
    w = 32  # Layers width
    kn = 64  # Number of filters
    kh = 3  # Filters weights height
    kw = 3  # Filters weights width
    vpadding = 1  # Vertical padding
    hpadding = 1  # Horizontal padding
    vstride = 2  # Vertical stride
    hstride = 2  # Horizontal stride
    # Create weights, x, and biases matrices from previous parameters. If no biases
    # matrix is provided, a proper one filled with zeros will be automatically
    # created.
    weights = np.zeros((kn, c, kh, kw)).astype(np.float32, order='C')
    weights[0][0][0][0] = 1.89
    weights[1][1][1][1] = 3.0
    weights[2][2][2][2] = 4.0
    x = np.ones((b, c, h, w)).astype(np.float32, order='C')
    ho = (h + 2 * vpadding - kh) // vstride + 1
    wo = (w + 2 * hpadding - kw) // hstride + 1
    biases = (np.ones((kn, b * ho * wo)) * 10).astype(np.float32, order='C')
    print("Using conv_gemm to compute alpha * weights * im2col(x) + beta * biases...")
    conv_gemm = ConvGemm(debug=False)
    conv_gemm_result = conv_gemm.conv_gemm(weights, x,
                                           vpadding=vpadding, hpadding=hpadding,
                                           vstride=vstride, hstride=hstride)
    print(conv_gemm_result)
    print("Sum: ", conv_gemm_result.sum())
    conv_gemm_t = timeit(lambda: conv_gemm.conv_gemm(weights, x,
                                                     vpadding=vpadding, hpadding=hpadding,
                                                     vstride=vstride, hstride=hstride),
                         number=10) / 10
    print("conv_gemm time: {:.4f}".format(conv_gemm_t))
    print()
    print("Using im2col and mm...")
    x_c = im2col_cython(x, kh, kw, vpadding, hpadding, vstride, hstride)
    w_c = weights.reshape(kn, -1)
    im2col_mm_result = w_c @ x_c + biases
    print(im2col_mm_result)
    print("Sum: ", im2col_mm_result.sum())
    print("np.allclose: ", np.allclose(conv_gemm_result, im2col_mm_result))
    im2col_t = timeit(lambda: im2col_cython(x, kh, kw, vpadding, hpadding, vstride, hstride), number=10) / 10
    print("im2col time: {:.4f}".format(im2col_t))
    mm_t = timeit(lambda: w_c @ x_c + biases, number=10) / 10
    print("mm time: {:.4f}".format(mm_t))


if __name__ == "__main__":
    __usage_example__()
