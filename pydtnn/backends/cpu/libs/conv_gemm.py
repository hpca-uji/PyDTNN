"""
PyDTNN convGemm module
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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import ctypes
import platform
import weakref

import numpy as np

from pydtnn.utils import load_library

try:
   load_library("convGemm")
   is_conv_gemm_available = True
except ImportError:
   is_conv_gemm_available = False


class ConvGemm:
    """
    Exposes the libconvGemm functions following the PyDTNN conventions.

    Methods
    -------
    conv_gemm(weights, x, biases, alpha, beta, vpadding, hpadding, vstride, hstride,
              vdilation, hdilation, biases_vector)
        Calls the appropriate convGemm function from libconvGemm.so to perform a
        matrix matrix multiplication with an implicit im2col.

    Examples
    --------
    See __usage_example__() method for an example of use. This example can be
    run with: 'python conv_gemm.py'

    Tests
    -----
    To perform the tests, run the following command from the current directory:
        python -m unittest tests.ConvGemmTestCase

    (see tests/conv_gemm.py for more instructions on testing)
    """

    lib_cg = None  # will link to the libconvGemm.so library

    def __init__(self, m=0, n=0, k=0, dtype=np.float32, debug=False, parent_layer=None):
        """
        Loads the libconvGemm.so library and creates the required auxiliary matrices ac_pack and bc_pack.

        Parameters
        ----------
        dtype : data type
            The element data type being used on all the matrices.
        debug : boolean
            Whether to print debug information or not.
        parent_layer: layer
            The layer that is using it (for tracing purposes).
        """
        if isinstance(dtype, type):
            self.dtype = dtype
        else:
            try:
                self.dtype = {'float32': np.float32, 'float64': np.float64}[dtype]
            except KeyError:
                raise AttributeError("dtype '{}' not recognized".format(dtype)) from None
        if ConvGemm.lib_cg is None:
            ConvGemm.lib_cg = load_library("convGemm")

        # Declare ac_pack and bc_pack and allocate space for them
        self.ac_pack = ctypes.POINTER(ctypes.c_float)()
        self.bc_pack = ctypes.POINTER(ctypes.c_float)()
        self.cc_pack = ctypes.POINTER(ctypes.c_float)()
        self.lib_cg.alloc_pack_buffs.restype = ctypes.c_int
        result = self.lib_cg.alloc_pack_buffs(
                ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
                ctypes.byref(self.ac_pack), ctypes.byref(self.bc_pack), ctypes.byref(self.cc_pack))
        if result == 1:
            raise MemoryError("Could not allocate space for ac_pack or bc_pack!")
        # Debug
        self.debug = debug
        # Parent layer
        if parent_layer is not None:
            self.get_parent_layer = weakref.ref(parent_layer)
        # Choose the appropriate convGemm function depending on the architecture and the data type being used
        if self.dtype == np.float32:
            self.x_conv_gemm_nhwc = self.lib_cg.sconvGemmNHWC
            self.x_deconv_gemm_nhwc = self.lib_cg.sconvGemmNHWC_back
            self.x_conv_gemm_nchw = self.lib_cg.sconvGemmNCHW
            self.x_deconv_gemm_nchw = self.lib_cg.sconvGemmNCHW_back
        else:
            raise ValueError("Type {} not supported by this version of libconvGemm!".format(str(self.dtype)))

    def __del__(self):
        """Free the allocated matrices"""
        try:
            __free__(self.ac_pack)
            __free__(self.bc_pack)
            __free__(self.cc_pack)
        except AttributeError:
            pass

    def conv_gemm_nchw(self, weights, x, biases=None, alpha=1.0, beta=0.0, vpadding=0, hpadding=0, vstride=1, hstride=1,
                  vdilation=1, hdilation=1, biases_vector=None, trans=False):
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
            An optional biases matrix (kn x b*ho*wo). If provided, can be overwritten.
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
        vdilation : int
            The vertical dilation.
        hdilation : int
            The horizontal dilation.
        biases_vector: array_like
            The biases that have to be summed to all the elements in each output channel.
        trans: bool
            Perform the im2col(x) if False, or the im2colT(x) if True.

        Returns
        -------
        array_like
            The result of alpha * weights * im2col(x_padded) + beta * biases.
        """

        # Get matrices dimensions
        b, c, h, w = x.shape
        if not trans:
            kn, ck, kh, kw = weights.shape
            if biases is None:
                beta = 0.0
                ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
                wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
                biases = np.empty((b, kn, ho, wo), weights.dtype, order="C")
            else:
                bb, ho, wo, knb = biases.shape
                assert kn == knb, "Number of filters must be the same!"
                assert b == bb, "Batch size must be the same!"
        else:
            assert biases is not None, "If using the transposed convGemm, the biases matrix must be supplied"
            kn, ck, kh, kw = biases.shape
            bw, knw, ho, wo = weights.shape
            assert kn == knw, "Number of filters must be the same!"
            assert b == bw, "Batch size must be the same!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        # Check that dtype is the same on all the matrices
        assert weights.dtype == x.dtype == biases.dtype, \
            "All the matrices must have the same type of data!"
        assert weights.dtype == self.dtype, \
            "The input matrices must have the same type of data as the one specified when " \
            "this class was instantiated!"

        # Call the appropriate convGemm function from libconvGemm
        self.x_conv_gemm_nchw(ctypes.c_char(b'Y' if trans else b'N'),
                         ctypes.c_uint(b), ctypes.c_uint(c), ctypes.c_uint(h), ctypes.c_uint(w),
                         ctypes.c_uint(kn), ctypes.c_uint(kh), ctypes.c_uint(kw),
                         ctypes.c_uint(vpadding), ctypes.c_uint(hpadding),
                         ctypes.c_uint(vstride), ctypes.c_uint(hstride),
                         ctypes.c_uint(vdilation), ctypes.c_uint(hdilation),
                         ctypes.c_float(alpha), ctypes.c_void_p(weights.ctypes.data),
                         ctypes.c_void_p(x.ctypes.data), ctypes.c_float(beta),
                         ctypes.c_void_p(biases.ctypes.data),
                         ctypes.c_void_p(None if biases_vector is None else biases_vector.ctypes.data),
                         self.ac_pack, self.bc_pack, self.cc_pack)

        return biases

    def conv_gemm_nhwc(self, weights, x, biases=None, alpha=1.0, beta=0.0, vpadding=0, hpadding=0, vstride=1, hstride=1,
                  vdilation=1, hdilation=1, biases_vector=None, trans=False):

        b, h, w, c = x.shape

        if not trans:
            ck, kh, kw, kn = weights.shape
            if biases is None:
                beta = 0.0
                ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
                wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
                biases = np.empty((b, ho, wo, kn), weights.dtype, order="C")
            else:
                bb, ho, wo, knb = biases.shape
                assert kn == knb, "Number of filters must be the same!"
                assert b == bb, "Batch size must be the same!"
        else:
            assert biases is not None, "If using the transposed convGemm, the output matrix must be supplied"
            ck, kh, kw, kn = biases.shape
            bw, ho, wo, knw  = weights.shape
            assert kn == knw, "Number of filters must be the same!"
            assert b == bw, "Batch size must be the same!"

        assert ck == c, "Number of channels in weights and x should be the same!"

        self.x_conv_gemm_nhwc(ctypes.c_char(b'Y' if trans else b'N'),
                         ctypes.c_uint(b), ctypes.c_uint(h), ctypes.c_uint(w), ctypes.c_uint(c),
                         ctypes.c_uint(kn), ctypes.c_uint(kh), ctypes.c_uint(kw),
                         ctypes.c_uint(vpadding), ctypes.c_uint(hpadding),
                         ctypes.c_uint(vstride), ctypes.c_uint(hstride),
                         ctypes.c_uint(vdilation), ctypes.c_uint(hdilation),
                         ctypes.c_float(alpha), ctypes.c_void_p(weights.ctypes.data),
                         ctypes.c_void_p(x.ctypes.data), ctypes.c_float(beta),
                         ctypes.c_void_p(biases.ctypes.data),
                         ctypes.c_void_p(None if biases_vector is None else biases_vector.ctypes.data),
                         self.ac_pack, self.bc_pack, self.cc_pack)

        return biases

    def deconv_gemm_nchw(self, weights, dy, dx, alpha=1.0, vpadding=0, hpadding=0,
                    vstride=1, hstride=1, vdilation=1, hdilation=1):
        """
        Calls the appropriate deconv_gemm function from libconvGemm.so to perform
        an inplace matrix matrix multiplication and deconvolution:

            dx = col2im(alpha * weights_2D_T * dy_2D),

        where:
          * weights_2D_T is the weights matrix reshaped to 2D and transposed (c·kh·kw x kn),
          * dy_2D is the dy matrix transposed_1023 and reshaped to 2D (kn x b·ho·wo).

        Parameters
        ----------
        weights : array_like
            The weights matrix (kn x c x kh x kw).
        dy : array_like
            The dy matrix (b x kn x ho x wo).
        dx : array_like
            An empty dx matrix (b x c x h x w) that will be overwritten with col2im(alpha * weights_2D_T * dy_2D).
        alpha : float
            The alpha factor.
        vpadding : int
            The vertical padding to be applied to the x matrix.
        hpadding : int
            The horizontal padding to be applied to the x matrix.
        vstride : int
            The vertical stride.
        hstride : int
            The horizontal stride.
        vdilation : int
            The vertical dilation.
        hdilation : int
            The horizontal dilation.

        Returns
        -------
        array_like
            The dx matrix.
        """

        # Get matrices dimensions
        kn, ck, kh, kw = weights.shape
        b2, kn2, ho, wo = dy.shape
        b, c, h, w = dx.shape
        assert kn == kn2, "Number of filters outputs in weights and dy should be the same!"
        assert b == b2, "Different batch size!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        self.x_deconv_gemm_nchw(ctypes.c_uint(b), ctypes.c_uint(c), ctypes.c_uint(h), ctypes.c_uint(w),
                           ctypes.c_uint(kn), ctypes.c_uint(kh), ctypes.c_uint(kw),
                           ctypes.c_uint(vstride), ctypes.c_uint(hstride),
                           ctypes.c_uint(vpadding), ctypes.c_uint(hpadding),
                           ctypes.c_uint(vdilation), ctypes.c_uint(hdilation),
                           ctypes.c_float(alpha), ctypes.c_void_p(weights.ctypes.data),
                           ctypes.c_void_p(dy.ctypes.data),
                           ctypes.c_void_p(dx.ctypes.data),
                           self.ac_pack, self.bc_pack, self.cc_pack)

        return dx

    def deconv_gemm_nhwc(self, weights, dy, dx, alpha=1.0, vpadding=0, hpadding=0,
                    vstride=1, hstride=1, vdilation=1, hdilation=1):

        ck, kh, kw, kn = weights.shape
        b2, ho, wo, kn2 = dy.shape
        b, h, w, c = dx.shape
        assert kn == kn2, "Number of filters outputs in weights and dy should be the same!"
        assert b == b2, "Different batch size!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        self.x_deconv_gemm_nhwc(ctypes.c_uint(b), ctypes.c_uint(h), ctypes.c_uint(w), ctypes.c_uint(c),
                        ctypes.c_uint(kn), ctypes.c_uint(kh), ctypes.c_uint(kw),
                        ctypes.c_uint(vstride), ctypes.c_uint(hstride),
                        ctypes.c_uint(vpadding), ctypes.c_uint(hpadding),
                        ctypes.c_uint(vdilation), ctypes.c_uint(hdilation),
                        ctypes.c_float(alpha), ctypes.c_void_p(weights.ctypes.data),
                        ctypes.c_void_p(dy.ctypes.data), ctypes.c_void_p(dx.ctypes.data),
                        self.ac_pack, self.bc_pack, self.cc_pack)

        return dx

def __free__(pack):
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
    assert isinstance(pack, object)
    libc.free(pack)


def __usage_example__():
    # Imports for this usage example (not required otherwise)
    from timeit import timeit
    from pydtnn.cython_modules import im2col_nchw_cython
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
    vdilation = 1  # Vertical dilation
    hdilation = 1  # Horizontal dilation
    # Create weights, x, and biases matrices from previous parameters. If no biases
    # matrix is provided, a proper one filled with zeros will be automatically
    # created.
    weights = np.zeros((kn, c, kh, kw)).astype(np.float32, order='C')
    weights[0][0][0][0] = 1.89
    weights[1][1][1][1] = 3.0
    weights[2][2][2][2] = 4.0
    x = np.ones((b, c, h, w)).astype(np.float32, order='C')
    ho = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    wo = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
    biases = (np.ones((kn, b * ho * wo)) * 10).astype(np.float32, order='C')
    print("Using conv_gemm to compute alpha * weights * im2col(x) + beta * biases...")
    conv_gemm = ConvGemm(debug=False)
    conv_gemm_result = conv_gemm.conv_gemm_nchw(weights, x,
                                           vpadding=vpadding, hpadding=hpadding,
                                           vstride=vstride, hstride=hstride,
                                           vdilation=vdilation, hdilation=hdilation)
    print(conv_gemm_result)
    print("Sum: ", conv_gemm_result.sum())
    conv_gemm_t = timeit(lambda: conv_gemm.conv_gemm_nchw(weights, x,
                                                     vpadding=vpadding, hpadding=hpadding,
                                                     vstride=vstride, hstride=hstride,
                                                     vdilation=vdilation, hdilation=hdilation),
                         number=10) / 10
    print("conv_gemm time: {:.4f}".format(conv_gemm_t))
    print()
    print("Using im2col and mm...")
    x_c = im2col_nchw_cython(x, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation)
    w_c = weights.reshape(kn, -1)
    im2col_mm_result = w_c @ x_c + biases
    print(im2col_mm_result)
    print("Sum: ", im2col_mm_result.sum())
    print("np.allclose: ", np.allclose(conv_gemm_result, im2col_mm_result))
    im2col_t = timeit(lambda: im2col_nchw_cython(x, kh, kw, vpadding, hpadding, vstride, hstride,
                                                 vdilation, hdilation), number=10) / 10
    print("im2col time: {:.4f}".format(im2col_t))
    mm_t = timeit(lambda: w_c @ x_c + biases, number=10) / 10
    print("mm time: {:.4f}".format(mm_t))


if __name__ == "__main__":
    __usage_example__()
