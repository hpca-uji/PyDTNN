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
from ctypes.util import find_library

import numpy as np


class ConvGemm:
    """
    Exposes the libconvGemm functions following the PyDTNN conventions.

    Methods
    -------
    conv_gemm(weights, x, biases, alpha, beta, vpadding, hpadding, vstride, hstride)
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
        self.lib = ctypes.cdll.LoadLibrary(path)
        if isinstance(dtype, type):
            self.dtype = dtype
        else:
            try:
                self.dtype = {'float32': np.float32, 'float64': np.float64}[dtype]
            except KeyError:
                raise AttributeError("dtype '{}' not recognized".format(dtype)) from None
        # Declare ac_pack and bc_pack and allocate space for them
        # @todo: The next fragment of code should be dependant on the architecture and the dtype
        self.ac_pack = ctypes.POINTER(ctypes.c_float)()
        self.bc_pack = ctypes.POINTER(ctypes.c_float)()
        self.lib.alloc_pack_buffs.restype = ctypes.c_int
        result = self.lib.alloc_pack_buffs(ctypes.byref(self.ac_pack), ctypes.byref(self.bc_pack))
        if result == 1:
            raise MemoryError("Could not allocate space for ac_pack or bc_pack!")
        self.debug = debug
        # Choose the appropriate convGemm function depending on the architecture and the data type being used
        if platform.machine() == 'aarch64':
            if self.dtype == np.float16:
                self.xconv_gemm = self.lib.sconvGemm
            elif self.dtype == np.float32:
                self.xconv_gemm = self.lib.hconvGemm
            else:
                raise ValueError("Type {} not supported by this version of libconvGemm!".format(str(self.dtype)))
        elif platform.machine() == 'x86_64':
            if self.dtype == np.float32:
                self.xconv_gemm = self.lib.sconvGemm
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

    def conv_gemm(self, weights, x, biases=None, alpha=1.0, beta=1.0, vpadding=0, hpadding=0, vstride=1, hstride=1):
        """
        Calls the appropriate convGemm function from libconvGemm.so to perform a
        matrix matrix multiplication with an implicit im2col.

        The matrix matrix product is in the form C = alpha * A * B + beta * C, where:
            + A is the weights matrix,
            + B is the im2col(x) matrix, and
            + C is the biases matrix.

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

        Returns
        -------
        array_like
            The result of alpha * weights * im2col(x_padded) + beta * biases.
        """

        # Pad x matrix and set vpadding and hpadding to 0
        x_padded = np.pad(x, ((0, 0), (0, 0), (vpadding, vpadding), (hpadding, hpadding)), mode='constant') \
            .astype(x.dtype)
        vpadding = hpadding = 0

        # Get matrices dimensions (once x matrix has been padded)
        kn, ck, kh, kw = weights.shape
        b, c, h, w = x_padded.shape
        assert ck == c, "Number of channels in weights and x should be the same!"

        # Compute height and weight of the output
        # Note: h and w are obtained from x_padded (no from x) and vpadding and
        #       hpadding are set to 0 in order to use the usual formulation
        ho = int((h + 2 * vpadding - kh) / vstride + 1)
        wo = int((w + 2 * hpadding - kw) / hstride + 1)

        # Create zero biases matrix if none provided. Otherwise, test its dimensions
        if biases is None:
            beta = 0.0
            biases = np.zeros((kn, b * ho * wo)).astype(weights.dtype, order='C')
        else:
            biases = biases.copy()  # To avoid overriding the original biases matrix
            assert (kn, b * ho * wo) == biases.shape, \
                "Biases matrix should be ({}, {}), instead it is {}".format(kn, b * ho * wo, biases.shape)

        # Check that dtype is the same on all the matrices
        assert weights.dtype == x.dtype == biases.dtype, \
            "All the matrices must have the same type of data!"
        assert weights.dtype == self.dtype, \
            "The input matrices must have the same type of data as the one specified when " \
            "this class was instantiated!"

        # Change matrices axes to the convGemm expected order:
        #   Where I→hi×wi×ci×b corresponds to the input tensor,
        #   F→kn×kh×kw×ci denotes the filters,
        #   and O→kn×ho×wo×b is the output tensor
        weights_cg = weights.transpose((0, 2, 3, 1)).reshape((kn, -1), order="F")
        x_padded_cg = x_padded.transpose((2, 3, 1, 0)).flatten(order="F")
        biases_cg = biases.astype(weights.dtype, order="F")

        # Call custom added function to libconvGemm.so to print the received parameters
        if self.debug:
            try:
                self.lib.expose_sconvGemm(ctypes.c_uint(kh), ctypes.c_uint(kw),
                                          ctypes.c_uint(c), ctypes.c_uint(kn),
                                          ctypes.c_float(alpha), ctypes.c_void_p(weights_cg.ctypes.data),
                                          ctypes.c_uint(h), ctypes.c_uint(w),
                                          ctypes.c_uint(b), ctypes.c_uint(vstride), ctypes.c_uint(hstride),
                                          ctypes.c_void_p(x_padded_cg.ctypes.data), ctypes.c_float(beta),
                                          ctypes.c_void_p(biases_cg.ctypes.data),
                                          self.ac_pack, self.bc_pack)
            except AttributeError:
                print("Warning: Custom 'expose_sconvGemm' function is not present in 'libconvGemm.so'. "
                      "You can safely ignore this warning.")

        # Call appropriate convGemm function from libconvGemm
        self.xconv_gemm(ctypes.c_uint(kh), ctypes.c_uint(kw),
                        ctypes.c_uint(c), ctypes.c_uint(kn),
                        ctypes.c_float(alpha), ctypes.c_void_p(weights_cg.ctypes.data),
                        ctypes.c_uint(h), ctypes.c_uint(w),
                        ctypes.c_uint(b),  ctypes.c_uint(vstride), ctypes.c_uint(hstride),
                        ctypes.c_void_p(x_padded_cg.ctypes.data), ctypes.c_float(beta),
                        ctypes.c_void_p(biases_cg.ctypes.data),
                        self.ac_pack, self.bc_pack)

        # Change output matrix axes to the PyDTNN expected order:
        out = biases_cg.reshape((kn, b, wo, ho)).transpose((0, 1, 3, 2)).reshape(kn, -1, order="C")
        return out


def __usage_example__():
    # Imports for this usage example (not required otherwise)
    from timeit import timeit
    from NN_im2col_cython import im2col_cython
    # Default parameters
    b = 32  # Batch size
    c = 3  # Channels per layer
    h = 128  # Layers height
    w = 100  # Layers width
    kn = 8  # Number of filters
    kh = 16  # Filters weights height
    kw = 10  # Filters weights width
    vpadding = 1  # Vertical padding
    hpadding = 2  # Horizontal padding
    vstride = 1  # Vertical stride
    hstride = vstride  # Horizontal stride (convGemm does not support different vertical and horizontal strides)
    # Create weights, x, and biases matrices from previous parameters. If no biases
    # matrix is provided, a proper one filled with zeros will be automatically
    # created.
    weights = np.zeros((kn, c, kh, kw)).astype(np.float32, order='C')
    weights[0][0][0][0] = 1.89
    weights[1][1][1][1] = 3.0
    weights[2][2][2][2] = 4.0
    x = np.ones((b, c, h, w)).astype(np.float32, order='C')
    ho = int((h + 2 * vpadding - kh) / vstride + 1)
    wo = int((w + 2 * hpadding - kw) / hstride + 1)
    biases = (np.ones((kn, b * ho * wo)) * 10).astype(np.float32, order='C')
    print("Using conv_gemm to compute alpha * weights * im2col(x) + beta * biases...")
    conv_gemm = ConvGemm(debug=False)
    conv_gemm_result = conv_gemm.conv_gemm(weights, x, biases=biases,
                                           vpadding=vpadding, hpadding=hpadding,
                                           vstride=vstride, hstride=hstride)
    print(conv_gemm_result)
    print("Sum: ", conv_gemm_result.sum())
    sconv_gemm_t = timeit(lambda: conv_gemm.conv_gemm(weights, x, biases=biases,
                                                      vpadding=vpadding, hpadding=hpadding,
                                                      vstride=vstride, hstride=hstride),
                          number=5) / 5

    print("conv_gemm time: {:.2f}".format(sconv_gemm_t))
    print()
    print("Using im2col and mm...")
    x_c = im2col_cython(x, kh, kw, vpadding, hpadding, vstride, hstride)
    w_c = weights.reshape(kn, -1)
    im2col_mm_result = w_c @ x_c + biases
    print(im2col_mm_result)
    print("Sum: ", im2col_mm_result.sum())
    print("np.allclose: ", np.allclose(conv_gemm_result, im2col_mm_result))
    im2col_t = timeit(lambda: im2col_cython(x, kh, kw, vpadding, hpadding, vstride, hstride), number=5) / 5
    print("im2col time: {:.2f}".format(im2col_t))
    mm_t = timeit(lambda: w_c @ x_c + biases, number=5) / 5
    print("mm time: {:.2f}".format(mm_t))


if __name__ == "__main__":
    __usage_example__()
