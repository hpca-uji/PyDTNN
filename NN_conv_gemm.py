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
import platform
import weakref
from contextlib import suppress

import numpy as np

from NN_tracer import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_BACKWARD_DCG_TRANSPOSE_DY, \
    PYDTNN_OPS_BACKWARD_DCG_SHRINK, PYDTNN_OPS_CONVGEMM_CG, PYDTNN_OPS_CONVGEMM_X_PAD, PYDTNN_OPS_CONVGEMM_TRANS_X_PAD, \
    PYDTNN_OPS_CONVGEMM_TRANS_CG, PYDTNN_OPS_CONVGEMM_TRANS_TR1230, PYDTNN_OPS_CONVGEMM_TRANS_BIASES
from NN_transpose_cython import transpose_1230_ji_cython, transpose_0231_kji_cython
from NN_util import load_library


class ConvGemmCache(dict):
    """
    Dictionary derived class that can use the provided factory function to
    obtain a default value for a missing key. It differs from defaultdict in:

    * The provided factory function receives key as a parameter (which allows
      the generated value to depend on the given key).

    * If disable() is called, the instances of this class will clear their
      already stored values and will not store the next ones.

    """
    _preserve_values = True

    def __init__(self, default_factory=None, **kwargs):
        super().__init__(self, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self.default_factory(key)
            if self._preserve_values:
                self[key] = ret
            return ret

    @classmethod
    def disable(cls):
        cls._preserve_values = False
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                obj.clear()


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
    dy_cg_cache = None
    dx_padded_cache = None

    # out_cg_cache = None  # Warning: don't use an static cached matrix for the output. No, do not do it.

    def __init__(self, dtype=np.float32, debug=False, parent_layer=None):
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
            ConvGemm.biases_cg_cache = ConvGemmCache(lambda shape: np.empty(shape, self.dtype, order="F"))
            ConvGemm.weights_cg_cache = ConvGemmCache(lambda shape: np.empty(shape, self.dtype, order="C"))
            ConvGemm.dy_cg_cache = ConvGemmCache(lambda shape: np.empty(shape, self.dtype, order="C"))
            ConvGemm.dx_padded_cache = ConvGemmCache(lambda shape: np.empty(shape, self.dtype, order="C"))
        # The x_padded and output matrices are also cached, but only on the current instance
        self.x_padded_cache = ConvGemmCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.out_cg_cache = ConvGemmCache(lambda shape: np.empty(shape, self.dtype, order="C"))

        # Declare ac_pack and bc_pack and allocate space for them
        # @todo: The next fragment of code should be dependant on the architecture and the dtype
        self.ac_pack = ctypes.POINTER(ctypes.c_float)()
        self.bc_pack = ctypes.POINTER(ctypes.c_float)()
        self.lib_cg.alloc_pack_buffs.restype = ctypes.c_int
        result = self.lib_cg.alloc_pack_buffs(ctypes.byref(self.ac_pack), ctypes.byref(self.bc_pack))
        if result == 1:
            raise MemoryError("Could not allocate space for ac_pack or bc_pack!")
        # Declare cc_pack
        self.cc_pack = ctypes.POINTER(ctypes.c_float)()
        self._cc_pack_size = 0
        # Debug
        self.debug = debug
        # Parent layer
        if parent_layer is not None:
            self.get_parent_layer = weakref.ref(parent_layer)
        # Choose the appropriate convGemm function depending on the architecture and the data type being used
        if platform.machine() == 'aarch64':
            if self.dtype == np.float16:
                self.x_conv_gemm = self.lib_cg.hconvGemm
            elif self.dtype == np.float32:
                self.x_conv_gemm = self.lib_cg.sconvGemm
                self.x_apply_bias = self.lib_cg.sapplyBias
                self.x_deconv_gemm = self.lib_cg.sconvGemm_back
            else:
                raise ValueError("Type {} not supported by this version of libconvGemm!".format(str(self.dtype)))
        elif platform.machine() == 'x86_64':
            if self.dtype == np.float32:
                self.x_conv_gemm = self.lib_cg.sconvGemm
                self.x_apply_bias = self.lib_cg.sapplyBias
                self.x_deconv_gemm = self.lib_cg.sconvGemm_back
            else:
                raise ValueError("Type {} not supported by this version of libconvGemm!".format(str(self.dtype)))
        else:
            raise ValueError("Platform '{}' not yet supported")

    def _alloc_cc_pack(self, kh, kw, c):
        size = kh * kw * c
        if self._cc_pack_size == size:  # Already allocated, nothing to be done
            return
        if self._cc_pack_size != 0:  # Free currently allocated memory
            __free__(self.cc_pack)
            self._cc_pack_size = 0
        # int alloc_unpack_buff(unsigned int kh, unsigned int kw, unsigned int c,float** Cc_pack)
        result = self.lib_cg.alloc_unpack_buff(ctypes.c_int(kh), ctypes.c_int(kw), ctypes.c_int(c),
                                               ctypes.byref(self.cc_pack))
        if result == 1:
            raise MemoryError("Could not allocate space for cc_pack!")
        self._cc_pack_size = size

    def __del__(self):
        """Free the allocated matrices"""
        try:
            __free__(self.ac_pack)
            __free__(self.bc_pack)
            if self._cc_pack_size != 0:
                __free__(self.cc_pack)
        except AttributeError:
            pass

    def conv_gemm(self, weights, x, biases=None, alpha=1.0, beta=1.0, vpadding=0, hpadding=0, vstride=1, hstride=1,
                  biases_vector=None, trans=False):
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
        biases_vector: array_like
            The biases that have to be summed to all the elements in each output channel.
        trans: bool
            Perform the im2col(x) if False, or the im2colT(x) if True.

        Returns
        -------
        array_like
            The result of alpha * weights * im2col(x_padded) + beta * biases.
        """

        # convGemm expected order for all matrices
        # ----------------------------------------
        #   Where I→hi×wi×ci×b corresponds to the input tensor, | PyDTNN (C order): b×ci×hi×wi   (F order: wi×hi×ci×b)
        #   F→kn×kh×kw×ci denotes the filters,                  | PyDTNN (C order): kn×ci×kh×kw  (F order: kw×kh×ci×kn)
        #   and O→kn×(ho·wo·b) is the output tensor             | PyDTNN (C order): kn×(b·ho·wo) (F order: (wo·ho·b)×kn)

        # --------------------------------------------------
        # 1) Pad x matrix and set vpadding and hpadding to 0
        # --------------------------------------------------
        if vpadding == 0 and hpadding == 0:
            x_padded = x
        else:
            with suppress(AttributeError):
                if not trans:
                    self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                              self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                              PYDTNN_OPS_CONVGEMM_X_PAD)
                else:
                    self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                              self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                              PYDTNN_OPS_CONVGEMM_TRANS_X_PAD)
            # Hand made alternative to:
            #  x_padded = np.pad(x, ((0, 0), (0, 0), (vpadding, vpadding), (hpadding, hpadding)), mode='constant')
            b, c, h, w = x.shape
            new_h, new_w = h + 2 * vpadding, w + 2 * hpadding
            # Padding alternative 1)
            x_padded = np.zeros((b, c, new_h, new_w), x.dtype)
            x_padded[:, :, vpadding:new_h - vpadding, hpadding:new_w - hpadding] = x
            # Padding alternative 2)
            # Better for matrices greater than a given size (but is machine dependant).
            # x_padded = self.x_padded_cache[(b, c, new_h, new_w)]
            # pad_cython(x, x_padded)
            vpadding = hpadding = 0
            with suppress(AttributeError):
                self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # ----------------------------------------------------------
        # 2) Get matrices dimensions (once x matrix has been padded)
        # ----------------------------------------------------------
        if not trans:
            kn, ck, kh, kw = weights.shape
            b, c, h, w = x_padded.shape
            assert ck == c, "Number of channels in weights and x should be the same!"
            # Compute height and weight of the output
            # Note: h and w are obtained from x_padded (no from x) and vpadding and
            #       hpadding were set to 0 just to use the usual formulation
            ho = (h + 2 * vpadding - kh) // vstride + 1
            wo = (w + 2 * hpadding - kw) // hstride + 1
        else:
            assert biases is not None, "If using the transposed convGemm, the biases matrix must be supplied"
            # if trans: Filters = Output_1023 x Im2ColT(Input)
            kn, ck, kh, kw = biases.shape  # Filters
            b, c, h, w = x_padded.shape  # Input
            knw, bw, ho, wo = weights.shape  # Output_1023
            assert ck == c, "Number of channels in biases and x should be the same!"
            assert kn == knw, "Number of filters in biases and weights must be the same!"
            assert b == bw, "Batch size in x and weights must be the same!"
            biases = biases.reshape((kn, -1), order="F")

        # -----------------------------------------------------------------------------
        # 3) Create zero biases matrix if none provided. Otherwise, test its dimensions
        # -----------------------------------------------------------------------------
        if trans:
            # Swap dimensions to regular usage
            kh, ho = ho, kh
            kw, wo = wo, kw
            b, c = c, b
            self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                      self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                      PYDTNN_OPS_CONVGEMM_TRANS_BIASES)
        if biases is None:
            # @warning: np.empty() could be used instead of np.zeros() only if the underlying operation does not
            # sum the biases when beta is 0.0. Otherwise, as the non initialized biases matrix could have nan elements,
            # the output will also have nan elements.
            beta = 0.0
            # biases_cg = np.empty((kn, b * ho * wo), weights.dtype, order='F')
            biases_cg = self.biases_cg_cache[(kn, b * ho * wo)]
        else:
            # biases = biases.copy()  # To avoid overriding the original biases matrix
            # Option 1) slow, even if same type and order
            #   biases_cg = biases.astype(weights.dtype, order="F")
            # Option 2) should be slower than the next one
            #   biases_cg = np.empty((kn, b * ho * wo), weights.dtype, order='F')
            #   biases_cg[...] = biases
            # Option 3)
            if biases.flags["F_CONTIGUOUS"]:
                biases_cg = biases
            else:
                biases_cg = biases.ravel(order="F").reshape(biases.shape, order="F")
            assert (kn, b * ho * wo) == biases.shape, \
                "Biases matrix should be ({}, {}), instead it is {}".format(kn, b * ho * wo, biases.shape)
        if trans:
            with suppress(AttributeError):
                self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # ---------------------------------------------------
        # 4) Check that dtype is the same on all the matrices
        # ---------------------------------------------------
        assert weights.dtype == x.dtype == biases_cg.dtype, \
            "All the matrices must have the same type of data!"
        assert weights.dtype == self.dtype, \
            "The input matrices must have the same type of data as the one specified when " \
            "this class was instantiated!"

        # -------------------------
        # 5) Transpose_1230 weights
        # -------------------------
        if trans:
            self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                      self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                      PYDTNN_OPS_CONVGEMM_TRANS_TR1230)
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
        # weights_cg = self.weights_cg_cache[(c, kh, kw, kn)]
        # weights_cg = np.empty((c, kh, kw, kn), weights.dtype, order="C")
        # assert weights.flags["C_CONTIGUOUS"] is True, \
        #     "sreshapeWeights_pydtnn does a bulk copy, the result will be wrong if weights is no C_CONTIGUOUS"
        # self.lib_cg.sreshapeWeights_pydtnn(ctypes.c_uint(kn), ctypes.c_uint(c), ctypes.c_uint(kw), ctypes.c_uint(kh),
        #                                    ctypes.c_void_p(weights.ctypes.data),
        #                                    ctypes.c_void_p(weights_cg.ctypes.data))
        # NEW(wxh): 6)
        # weights_cg = np.empty((c, kh, kw, kn), weights.dtype, order="C")
        weights_cg = self.weights_cg_cache[(c, kh, kw, kn)]
        transpose_1230_ji_cython(weights, weights_cg)
        if trans:
            with suppress(AttributeError):
                self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # -------------------
        # 6) Flatten x_padded
        # -------------------
        # PREVIOUS(hxw): x_padded_cg = x_padded.transpose((2, 3, 1, 0)).flatten(order="F")
        # NEW(wxh) 1): x_padded_cg = x_padded.transpose((3, 2, 1, 0)).flatten(order="F")
        # NEW(wxh) 2): x_padded_cg = x_padded.ravel(order="C")
        # NEW(wxh) 3):
        x_padded_cg = x_padded.ravel(order="K")

        # --------------------------------------------------------------------------------
        # 7) Call custom added function to libconvGemm.so to print the received parameters
        # --------------------------------------------------------------------------------
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

        # -------------------------------------------------------------------------------------
        # 8) Call the appropriate convGemm function from libconvGemm (height and width swapped)
        # -------------------------------------------------------------------------------------
        if trans:
            # Swap dimensions back to transposed usage
            kh, ho = ho, kh
            kw, wo = wo, kw
            b, c = c, b
        with suppress(AttributeError):
            if not trans:
                self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                          self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                          PYDTNN_OPS_CONVGEMM_CG)
            else:
                self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                          self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                          PYDTNN_OPS_CONVGEMM_TRANS_CG)
        self.x_conv_gemm(ctypes.c_char(b'Y' if trans else b'N'),
                         ctypes.c_uint(kw), ctypes.c_uint(kh),
                         ctypes.c_uint(c), ctypes.c_uint(kn),
                         ctypes.c_float(alpha), ctypes.c_void_p(weights_cg.ctypes.data),
                         ctypes.c_uint(w), ctypes.c_uint(h),
                         ctypes.c_uint(b), ctypes.c_uint(hstride), ctypes.c_uint(vstride),
                         ctypes.c_void_p(x_padded_cg.ctypes.data), ctypes.c_float(beta),
                         ctypes.c_void_p(biases_cg.ctypes.data),
                         self.ac_pack, self.bc_pack)
        with suppress(AttributeError):
            self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if trans:
            # Swap dimensions to regular usage
            kh, ho = ho, kh
            kw, wo = wo, kw
            b, c = c, b

        # ----------------------
        # 9) Apply biases vector
        # ----------------------
        if biases_vector is not None:
            self.x_apply_bias(ctypes.c_uint(kn), ctypes.c_uint(b * ho * wo),
                              ctypes.c_void_p(biases_vector.ctypes.data), ctypes.c_void_p(biases_cg.ctypes.data),
                              ctypes.c_uint(kn))

        # -----------------------------------------------------
        # 10) Change output matrix to the PyDTNN expected order
        # -----------------------------------------------------
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
        out = np.empty((kn, b * ho * wo), weights.dtype, order="C")
        # Warning: don't use a matrix cache for out (it will be rewritten elsewhere, no matter how hard you try not to)
        # out = self.out_cg_cache[(kn, b * ho * wo)]
        self.lib_cg.sreshapeOut_pydtnn(ctypes.c_uint(kn), ctypes.c_uint(b), ctypes.c_uint(wo), ctypes.c_uint(ho),
                                       ctypes.c_void_p(biases_cg.ctypes.data), ctypes.c_void_p(out.ctypes.data))
        # * NEW(wxh) 4):
        # out = np.empty((kn, b * ho * wo), weights.dtype, order="C")
        # transpose_2d_f2c_ji_cython(biases_cg, out)

        return out

    def deconv_gemm(self, weights, dy, dx, alpha=1.0, vpadding=0, hpadding=0, vstride=1, hstride=1):
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

        Returns
        -------
        array_like
            The dx matrix.
        """

        # Get matrices dimensions
        kn, ck, kh, kw = weights.shape
        b, kn2, ho, wo = dy.shape
        b, c, h, w = dx.shape
        assert kn == kn2, "Number of filters outputs in weights and dy should be the same!"
        assert ck == c, "Number of channels in weights and x should be the same!"

        # Allocate space for self.cc_pack
        self._alloc_cc_pack(kh, kw, c)

        # Change matrices axes to the convGemm expected order:
        #    W → kh*kw*c×kn  | PyDTNN (C order): kn×ci×kh×kw  (F order: kw×kh×ci×kn)
        #   DY → kn×ho*wo*b  | PyDTNN (C order): b×kn×ho×wo (F order: wo×ho×kn×b)
        #                                        0 1  2  3             3  2  1 0
        #        1  3  2  0  ->  0231
        dy_cg = self.dy_cg_cache[(b, ho, wo, kn)]
        with suppress(AttributeError):
            self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                      self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                      PYDTNN_OPS_BACKWARD_DCG_TRANSPOSE_DY)
        transpose_0231_kji_cython(dy, dy_cg)  # Faster than the ijk version
        with suppress(AttributeError):
            self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # void sconvGemm_back(unsigned int kh, unsigned int kw, unsigned int c, unsigned int kn,
        # 		              float alpha, float * A,
        #                     unsigned int h, unsigned int w, unsigned int b,
        #                     unsigned int hStride, unsigned int wStride,
        #                     unsigned int hPad, unsigned int wPad,
        # 		              float * B, float * C,
        #                     float * Ac_pack, float * Bc_pack, float* Cc_pack)

        # ---------------------.
        # Dx no padded version |
        # ---------------------·
        # # Call the appropriate deconvGemm function from libconvGemm (height and width swapped)
        # self.x_deconv_gemm(ctypes.c_uint(kw), ctypes.c_uint(kh),
        #                    ctypes.c_uint(c), ctypes.c_uint(kn),
        #                    ctypes.c_float(alpha), ctypes.c_void_p(weights.ctypes.data),
        #                    ctypes.c_uint(w), ctypes.c_uint(h), ctypes.c_uint(b),
        #                    ctypes.c_uint(hstride), ctypes.c_uint(vstride),
        #                    ctypes.c_uint(hpadding), ctypes.c_uint(vpadding),
        #                    ctypes.c_void_p(dy_cg.ctypes.data),
        #                    ctypes.c_void_p(dx.ctypes.data),
        #                    self.ac_pack, self.bc_pack, self.cc_pack)
        # return dx

        # ------------------.
        # Dx padded version |
        # ------------------·
        # dx_padded = np.empty((b, c, h + 2 * vpadding, w + 2 * hpadding), dtype=self.dtype)
        dx_padded = self.dx_padded_cache[(b, c, h + 2 * vpadding, w + 2 * hpadding)]
        # Call the appropriate deconvGemm function from libconvGemm (height and width swapped)
        self.x_deconv_gemm(ctypes.c_uint(kw), ctypes.c_uint(kh),
                           ctypes.c_uint(c), ctypes.c_uint(kn),
                           ctypes.c_float(alpha), ctypes.c_void_p(weights.ctypes.data),
                           ctypes.c_uint(w), ctypes.c_uint(h), ctypes.c_uint(b),
                           ctypes.c_uint(hstride), ctypes.c_uint(vstride),
                           ctypes.c_uint(hpadding), ctypes.c_uint(vpadding),
                           ctypes.c_void_p(dy_cg.ctypes.data),
                           ctypes.c_void_p(dx_padded.ctypes.data),
                           self.ac_pack, self.bc_pack, self.cc_pack)
        if vpadding or hpadding:
            with suppress(AttributeError):
                self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT,
                                                          self.get_parent_layer().id * PYDTNN_OPS_EVENTS +
                                                          PYDTNN_OPS_BACKWARD_DCG_SHRINK)
            dx[...] = dx_padded[:, :, vpadding:vpadding + h, hpadding:hpadding + w]
            with suppress(AttributeError):
                self.get_parent_layer().tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        else:
            dx = dx_padded
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
