"""
Layer definitions for Python Distributed Training of Neural Networks (PyDTNN)

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

import time
from contextlib import suppress
from functools import lru_cache
from math import floor

import NN_activation
import NN_initializer
from NN_add_cython import add_cython
from NN_argmax_cython import argmax_cython
from NN_base_layer import Layer
from NN_conv_gemm import ConvGemm, ConvGemmCache
from NN_im2col_cython import im2col_cython, col2im_cython
from NN_bn_inference_cython import bn_inference_cython
from NN_bn_relu_inference_cython import bn_relu_inference_cython
from NN_model import EVALUATE_MODE, TRAIN_MODE
from NN_pad_cython import pad_cython, transpose_1023_and_pad_cython
from NN_reindex_cython import reindex_cython
from NN_sim import *
from NN_tracer import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CONVGEMM, PYDTNN_OPS_FORWARD_RESHAPE_Y, \
    PYDTNN_OPS_COMP_DW_MATMUL, PYDTNN_OPS_COMP_DX_COL2IM, PYDTNN_OPS_COMP_DX_MATMUL, PYDTNN_OPS_FORWARD_IM2COL, \
    PYDTNN_OPS_BACKWARD_TRANSPOSE_DY, PYDTNN_OPS_BACKWARD_PADDING_X, \
    PYDTNN_OPS_BACKWARD_COMP_NEW_INDEXES, PYDTNN_OPS_BACKWARD_REINDEX, PYDTNN_OPS_BACKWARD_CONVGEMM, \
    PYDTNN_OPS_BACKWARD_SUM_BIASES, PYDTNN_OPS_FORWARD_MATMUL, PYDTNN_OPS_FORWARD_SUM_BIASES, \
    PYDTNN_OPS_FORWARD_RESHAPE_W, PYDTNN_OPS_BACKWARD_TRANSPOSE_W, PYDTNN_OPS_BACKWARD_RESHAPE_DW, \
    PYDTNN_OPS_BACKWARD_IM2COL, PYDTNN_OPS_BACKWARD_DECONV_GEMM, \
    PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_BACKWARD_ELTW_SUM, \
    PYDTNN_OPS_BACKWARD_SPLIT, PYDTNN_OPS_FORWARD_CONCAT, PYDTNN_OPS_FORWARD_ELTW_SUM, PYDTNN_OPS_FORWARD_REPLICATE

try:
    from mpi4py import MPI
    # import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    import libnccl.libnccl as nccl
except ModuleNotFoundError:
    pass
except ImportError:
    pass


# @todo: will be used when layer.initialize includes model: initialize(model, id, ...)
class ForwardToBackward:
    """
    Class used to store those items from the forward pass that are required on the backward pass. When the model
    is in evaluate mode, the passed items are not stored.
    """

    def __init__(self):
        self._model = None
        self._storage = {}

    def set_model(self, model):
        self._model = model

    def __setattr__(self, key, value):
        if self._model.mode == TRAIN_MODE:
            self._storage[key] = value
        else:
            if self._storage:
                self._storage.clear()

    def __getattr__(self, item):
        try:
            return self._storage[item]
        except KeyError:
            raise AttributeError from None


class Input(Layer):

    def __init__(self, shape=(1,)):
        super(Input, self).__init__(shape)


class FC(Layer):

    def __init__(self, shape=(1,), activation="", use_bias=True,
                 weights_initializer="glorot_uniform",
                 biases_initializer="zeros"):
        super(FC, self).__init__(shape)
        self.act = getattr(NN_activation, activation, None)
        self.use_bias = use_bias
        self.weights_initializer = getattr(NN_initializer, weights_initializer)
        self.biases_initializer = getattr(NN_initializer, biases_initializer)
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"
        # The next attributes will be initialized later
        self.x = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.weights = self.weights_initializer((*prev_shape, *self.shape), self.dtype)
        if self.use_bias:
            self.biases = self.biases_initializer(self.shape, self.dtype)
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)

        self.fwd_time = \
            matmul_time(m=self.batch_size, n=self.weights.shape[1], k=self.weights.shape[0],
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype)
        self.bwd_time = \
            matmul_time(m=self.weights.shape[0], n=self.weights.shape[1], k=self.batch_size,
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype) + \
            matmul_time(m=self.batch_size, n=self.weights.shape[0], k=self.weights.shape[1],
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype) if need_dx else 0

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.weights.shape), ""))

    def forward(self, x):
        if self.model.mode == TRAIN_MODE:
            self.x = x
        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_MATMUL)
        res = self.matmul(x, self.weights)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return res + self.biases if self.use_bias else 0

    def backward(self, dy):
        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DW_MATMUL)
        self.dw = self.matmul(self.x.T, dy)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.use_bias:
            self.db = np.sum(dy, axis=0)

        if self.need_dx:
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL)
            dx = self.matmul(dy, self.weights.T)
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx


class Conv2D(Layer):

    def __init__(self, nfilters=1, filter_shape=(3, 3), padding=0, stride=1,
                 activation="", use_bias=True, weights_initializer="glorot_uniform",
                 biases_initializer="zeros"):
        super(Conv2D, self).__init__()
        self.co = nfilters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        self.act = getattr(NN_activation, activation, None)
        self.use_bias = use_bias
        self.weights_initializer = getattr(NN_initializer, weights_initializer)
        self.biases_initializer = getattr(NN_initializer, biases_initializer)
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"
        self.debug = False
        # convGemm related attributes (some of them will be modified in initialize())
        self.cg = None
        self.cg_fallback_to_im2col = True  # Fallback to backward I2C if any stride is greater than 1
        self.cg_cache = True  # Store created matrices to allow them to be reused
        self.cg_deconv = False  # Use deconvGemm instead of matmul + col2im
        self.cg_trans = False  # Use the convGemm function to operate with im2colT
        self.cg_x_transposed_cache = ConvGemmCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.cg_x_indexed_cache = ConvGemmCache(lambda shape: np.zeros(shape, self.dtype, order="C"))
        self.cg_biases_cache = ConvGemmCache(lambda shape: np.empty(shape, self.dtype, order="F"))  # order F!
        self.cg_matmul_out_cache = ConvGemmCache(lambda shape: np.empty(shape, self.dtype, order="C"))
        # The next attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = 0
        # @warning: do not do this (affects the gpu version) self.forward = self.backward = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)

        self.ci, self.hi, self.wi = prev_shape
        self.kh, self.kw = self.filter_shape

        self.weights = self.weights_initializer(((self.co,) + (self.ci,) + self.filter_shape), self.dtype)
        if self.use_bias:
            self.biases = self.biases_initializer((self.co,), self.dtype)

        self.ho = floor((self.hi + 2 * self.vpadding - self.kh) / self.vstride) + 1
        self.wo = floor((self.wi + 2 * self.hpadding - self.kw) / self.hstride) + 1
        self.shape = (self.co, self.ho, self.wo)
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)

        if self.model.params.enable_conv_gemm:
            self.cg = ConvGemm(dtype=self.dtype, debug=self.debug, parent_layer=self)
            if not self.model.params.conv_gemm_cache:
                ConvGemmCache.disable()
            self.forward = self._forward_cg
            self.backward = self._backward_cg
            self.cg_fallback_to_im2col = self.model.params.conv_gemm_fallback_to_im2col
            self.cg_deconv = self.model.params.conv_gemm_deconv
            self.cg_trans = self.model.params.conv_gemm_trans
        else:
            self.forward = self._forward_i2c
            self.backward = self._backward_i2c

        if not self.debug:
            time.perf_counter = lambda: 0

        self.fwd_time = \
            im2col_time(m=(self.ci * self.kh * self.kw), n=(self.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype) + \
            matmul_time(m=self.co, n=(self.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype) + \
            matmul_time(m=(self.ci * self.kh * self.kw), n=(self.batch_size * self.ho * self.wo), k=self.co,
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype) if need_dx else 0 + \
                                                          col2im_time(m=(self.ci * self.kh * self.kw),
                                                                      n=(self.batch_size * self.ho * self.wo),
                                                                      cpu_speed=self.model.params.cpu_speed,
                                                                      memory_bw=self.model.params.memory_bw,
                                                                      dtype=self.dtype) if need_dx else 0

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.weights.shape), \
                                                "padd=(%d,%d), stride=(%d,%d)" % (
                                                    self.vpadding, self.hpadding, self.vstride, self.hstride)))

    def forward(self, x):
        """This is a fake forward function. It will be masked on initialization by _forward_i2c or _forward_cg"""
        pass

    def _forward_i2c(self, x):
        """Version of the forward function that uses im2col and matmul"""

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_cols = im2col_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.model.mode == TRAIN_MODE:
            self.x_cols = x_cols

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_W)
        w_cols = self.weights.reshape(self.co, -1)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_MATMUL)
        res = self.matmul(w_cols, x_cols)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_SUM_BIASES)
        # y = res + self.biases.reshape(-1, 1)
        y = add_cython(res, self.biases) if self.use_bias else res
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = y.reshape(self.co, -1, self.ho, self.wo).transpose(1, 0, 2, 3)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def _forward_cg(self, x):
        """Version of the forward function that uses the convGemm library"""

        if self.model.mode == TRAIN_MODE:
            self.cg_x = x

        biases_vector = self.biases if self.use_bias else None

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        res = self.cg.conv_gemm(self.weights, x, biases=None,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                biases_vector=biases_vector)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # Biases sum is now done on conv_gemm
        # y = res + self.biases.reshape(-1, 1)
        # y = add_cython(res, self.biases) if self.use_bias else res

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = res.reshape(self.co, -1, self.ho, self.wo).transpose(1, 0, 2, 3)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def backward(self, dy):
        """This is a fake backward function. It will be masked on initialization by _backward_i2c or _backward_cg"""
        pass

    def _backward_i2c(self, dy):
        """Version of the backward function that uses im2col and matmul"""
        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_DY)
        dy_cols = dy.transpose((1, 0, 2, 3)).reshape(self.co, -1)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_W)
        w_cols = self.weights.reshape(self.co, -1).T
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DW_MATMUL)
        res = self.matmul(dy_cols, self.x_cols.T)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_RESHAPE_DW)
        self.dw = res.reshape(self.weights.shape)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 2, 3))
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.need_dx:
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL)
            res = self.matmul(w_cols, dy_cols)
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = col2im_cython(res, dy.shape[0], self.ci, self.hi, self.wi,
                               self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx

    @staticmethod
    # @lru_cache(maxsize=4)
    def _get_x_new_indexes_and_xstride(kx, xo, s):
        """
        Returns x_reorder and xstride based on kx (kh or kw), xo (ho or wo), and s (hstride or
        vstride)
        """
        if s == 1:
            return None, 1
        x_reorder = []
        for i in range(kx):
            x_reorder += [i + j * s for j in range(xo)]
        # Return x_reorder as a numpy.array because indexing is faster with a numpy.array than with a list
        return np.array(x_reorder), xo

    def _backward_cg(self, dy):
        """Version of the backward function that uses the convGemm library"""

        # if self.id == 4:
        #     self.tracer.print_memory_usage(f"Inside layer {self.id:03} backward 01")

        if self.cg_fallback_to_im2col and (self.vstride > 1 or self.hstride > 1):
            #
            # Although it is is possible to reindex x so that convGemm library can be used to compute
            # dw = dy * im2col(x).T when any of the strides is greater than one, the cost of reindexing x is
            # considerable high. Therefore, it is possible to fall back to compute im2col(x) and to call the
            # original backward method.
            #
            # In order to use the convGemm library when any of the strides is greater than one:
            #  1) the matrix copy using the new indexes should be parallelized, or
            #  2) the underlying convGemm method should directly support the dy * im2col(x).T operation,
            #     where im2col(x) is the im2col(x) in weights * im2col(x) (not in dy * im2col(x)).
            #
            # As the first option has been implemented, using the convGemm library in this case is now competitive.
            #
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_IM2COL)
            self.x_cols = im2col_cython(self.cg_x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride)
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return self._backward_i2c(dy)

        if not self.cg_trans:
            # 1) cg_dy
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_DY)
            cg_dy = dy.transpose((1, 0, 2, 3))
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 2) cg_x_transposed
            b, c, h, w = self.cg_x.shape
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_PADDING_X)
            # if self.vpadding == 0 and self.hpadding == 0:
            #     self.cg_x_indexed = self.cg_x.transpose((1, 0, 2, 3))
            # else:
            #     # Hand made alternative to:
            #     # self.cg_x_indexed = np.pad(self.cg_x.transpose((1, 0, 2, 3)),
            #     #                            ((0, 0),
            #     #                             (0, 0),
            #     #                             (self.vpadding, self.vpadding),
            #     #                             (self.hpadding, self.hpadding)),
            #     #                            mode='constant')
            #     b, c, h, w = self.cg_x.shape
            #     new_h, new_w = h + 2 * self.vpadding, w + 2 * self.hpadding
            #     # self.cg_x_indexed = np.zeros((c, b, new_h, new_w), self.dtype)
            #     self.cg_x_indexed = self.cg_x_indexed_cache[(c, b, new_h, new_w)]
            #     self.cg_x_indexed[:, :, self.vpadding:new_h - self.vpadding, self.hpadding:new_w - self.hpadding] = \
            #         self.cg_x.transpose((1, 0, 2, 3))
            if self.vpadding == 0 and self.hpadding == 0:
                # @todo: cython transpose version
                cg_x_transposed = self.cg_x.transpose((1, 0, 2, 3))
            else:
                new_h, new_w = h + 2 * self.vpadding, w + 2 * self.hpadding
                cg_x_transposed = self.cg_x_transposed_cache[(c, b, new_h, new_w)]
                transpose_1023_and_pad_cython(self.cg_x, cg_x_transposed)
            cg_vpadding = 0
            cg_hpadding = 0
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 3) new indexes and strides
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_COMP_NEW_INDEXES)
            v_new_indexes, cg_vstride = self._get_x_new_indexes_and_xstride(self.kh, self.ho, self.vstride)
            h_new_indexes, cg_hstride = self._get_x_new_indexes_and_xstride(self.kw, self.wo, self.hstride)
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 4) cg_x_indexed
            # Indexing performance considerations:
            #  + Using numpy.array to select the indexes is faster than using a list
            #    (check _get_x_new_indexes_and_xstride).
            #  + Indexing first rows and then columns is faster than the opposite.
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_REINDEX)
            # if h_new_indexes is not None:
            #     self.cg_x_indexed = self.cg_x_indexed[:, :, h_new_indexes, :]
            # if v_new_indexes is not None:
            #     self.cg_x_indexed = self.cg_x_indexed[:, :, :, v_new_indexes]
            # if h_new_indexes is not None or v_new_indexes is not None:
            #     # @warning: The next line is required to ensure the correct order of the underlying data of
            #     #           self.cg_x_indexed. Otherwise using self.cg_x_indexed.ravel(order="K") will lead to
            #     #           unexpected results
            #     self.cg_x_indexed = self.cg_x_indexed.copy()
            if h_new_indexes is not None or v_new_indexes is not None:
                new_h = len(v_new_indexes) if v_new_indexes is not None else h
                new_w = len(h_new_indexes) if h_new_indexes is not None else w
                # self.cg_x_indexed = np.empty((c, b, new_h, new_w), dtype=self.dtype)
                cg_x_indexed = self.cg_x_indexed_cache[(c, b, new_h, new_w)]
                reindex_cython(v_new_indexes, h_new_indexes, cg_x_transposed, cg_x_indexed)
            else:
                cg_x_indexed = cg_x_transposed
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 5) cg_biases
            cg_biases = None
        else:
            # if self.cg_trans:
            # 1) cg_dy
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_DY)
            cg_dy = dy.transpose((1, 0, 2, 3))
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 2) cg_x_indexed
            cg_x_indexed = self.cg_x
            # 3) cg_biases
            cg_biases = self.cg_biases_cache[self.weights.shape]
            # 4) rest of parameters
            cg_vpadding = self.vpadding
            cg_hpadding = self.hpadding
            cg_vstride = self.vstride
            cg_hstride = self.hstride

        if self.debug:
            # Expose cg_x_indexed
            self.cg_x_indexed = cg_x_indexed

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CONVGEMM)
        res = self.cg.conv_gemm(cg_dy, cg_x_indexed,
                                biases=cg_biases, beta=0.0,
                                vpadding=cg_vpadding, hpadding=cg_hpadding,
                                vstride=cg_vstride, hstride=cg_hstride, trans=self.cg_trans)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_RESHAPE_DW)
        self.dw = res.reshape(self.weights.shape)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 2, 3))
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.need_dx:
            if self.cg_deconv:
                self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_DECONV_GEMM)
                dx = self.cg.deconv_gemm(self.weights, dy, self.cg_x,
                                         vpadding=self.vpadding, hpadding=self.hpadding,
                                         vstride=self.vstride, hstride=self.hstride)
                self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            else:
                w_cols = self.weights.reshape(self.co, -1).T
                dy_cols = cg_dy.reshape(self.co, -1)

                # Don't use a cached version if the res matrix will persist
                res = self.cg_matmul_out_cache[(w_cols.shape[0], dy_cols.shape[1])]

                self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL)
                self.matmul(w_cols, dy_cols, res)
                self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

                self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
                dx = col2im_cython(res, dy.shape[0], self.ci, self.hi, self.wi,
                                   self.kh, self.kw, self.vpadding, self.hpadding,
                                   self.vstride, self.hstride)
                self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx


class MaxPool2D(Layer):

    def __init__(self, pool_shape=(2, 2), padding=0, stride=1):
        super(MaxPool2D, self).__init__()
        self.pool_shape = pool_shape
        self.padding = padding
        self.stride = stride
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        # The next attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0
        self.maxids = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.ci, self.hi, self.wi = prev_shape
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.ho = (self.hi + 2 * self.vpadding - self.kh) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.kw) // self.hstride + 1
        self.co = self.ci
        self.shape = (self.co, self.ho, self.wo)
        self.n = np.prod(self.shape)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype) if need_dx else 0

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.pool_shape),
                                                "padd=(%d,%d), stride=(%d,%d)" %
                                                (self.vpadding, self.hpadding, self.vstride, self.hstride)))

    def forward(self, x):
        x_ = x.reshape(x.shape[0] * self.ci, 1, self.hi, self.wi)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_cols = im2col_cython(x_, self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # self.maxids = tuple([np.argmax(a_cols, axis=0), np.arange(a_cols.shape[1])])
        y, maxids = argmax_cython(x_cols, axis=0)

        if self.model.mode == TRAIN_MODE:
            self.maxids = maxids

        return y.reshape(x.shape[0], *self.shape)

    def backward(self, dy):
        if self.need_dx:
            dy_cols = np.zeros((self.kh * self.kw, np.prod(dy.shape)), dtype=self.dtype)
            dy_cols[self.maxids] = dy.flatten()
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = col2im_cython(dy_cols, dy.shape[0] * self.ci, 1, self.hi, self.wi,
                               self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            dx = dx.reshape(dy.shape[0], self.ci, self.hi, self.wi)
            return dx


class AveragePool2D(Layer):

    def __init__(self, pool_shape=(2, 2), padding=0, stride=1):
        super(AveragePool2D, self).__init__()
        self.pool_shape = pool_shape
        self.padding = padding
        self.stride = stride
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        # The next attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.ci, self.hi, self.wi = prev_shape
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.ho = (self.hi + 2 * self.vpadding - self.kh) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.kw) // self.hstride + 1
        self.co = self.ci
        self.shape = (self.co, self.ho, self.wo)
        self.n = np.prod(self.shape)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw,
                        dtype=self.dtype) if need_dx else 0

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.pool_shape), \
                                                "padd=(%d,%d), stride=(%d,%d)" % (
                                                    self.vpadding, self.hpadding, self.vstride, self.hstride)))

    def forward(self, x):
        x_ = x.reshape(x.shape[0] * self.ci, 1, self.hi, self.wi)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_cols = im2col_cython(x_, self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        y = np.mean(x_cols, axis=0)
        return y.reshape(x.shape[0], *self.shape)

    def backward(self, dy):
        if self.need_dx:
            pool_size = np.prod(self.pool_shape)
            dy_cols = np.tile(dy.flatten() / pool_size, (pool_size, 1))
            self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = col2im_cython(dy_cols, dy.shape[0] * self.ci, 1, self.hi, self.wi,
                               self.kh, self.kw, self.vpadding, self.hpadding,
                               self.vstride, self.hstride)
            self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            dx = dx.reshape(dy.shape[0], self.ci, self.hi, self.wi)
            return dx


class Dropout(Layer):

    def __init__(self, rate=0.5):
        super(Dropout, self).__init__()
        self.rate = min(1., max(0., rate))
        # The next attributes will be initialized later
        self.mask = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.shape = prev_shape

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format("", "rate=%.2f" % self.rate))

    def forward(self, x):
        if self.model.mode == TRAIN_MODE:
            self.mask = np.random.binomial(1, (1 - self.rate), size=self.shape).astype(self.dtype) / (1 - self.rate)
            return x * self.mask
        else:  # EVALUATE_MODE:
            return x

    def backward(self, dy):
        if self.need_dx:
            return dy * self.mask


class Flatten(Layer):

    def __init__(self):
        super(Flatten, self).__init__()

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.shape = (np.prod(prev_shape),)

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

    def backward(self, dy):
        if self.need_dx:
            return dy.reshape(dy.shape[0], *self.prev_shape)


class BatchNormalization(Layer):

    def __init__(self, beta=0.0, gamma=1.0,
                 momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 sync_stats=False):
        super(BatchNormalization, self).__init__()
        self.gamma_init_val = gamma
        self.beta_init_val = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = getattr(NN_initializer, moving_mean_initializer)
        self.moving_variance_initializer = getattr(NN_initializer, moving_variance_initializer)
        self.grad_vars = {"beta": "dbeta", "gamma": "dgamma"}
        self.sync_stats = sync_stats
        # The next attributes will be initialized later
        self.spatial = self.co = self.ci = self.hi = self.wi = 0
        self.gamma = self.beta = self.running_mean = self.running_var = None
        self.std = self.xn = None
        self.dgamma = self.dbeta = None
        self.updated_running_var = False

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.shape = shape_ = prev_shape
        self.spatial = len(self.shape) > 2
        if self.spatial:
            self.co = self.ci = self.shape[0]
            self.hi, self.wi = self.shape[1], self.shape[2]
            shape_ = (self.ci, )
        self.gamma = np.full(shape_, self.gamma_init_val, self.dtype)
        self.beta = np.full(shape_, self.beta_init_val, self.dtype)
        self.running_mean = self.moving_mean_initializer(shape_, self.dtype)
        self.running_var = self.moving_variance_initializer(shape_, self.dtype)
        self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)
        self.nparams = self.gamma.size + self.beta.size

    def forward(self, x):

        def mean(data, N, comm):
            if self.sync_stats and comm is not None:
                mean = np.sum(data, axis=0) / N
                comm.Allreduce(MPI.IN_PLACE, mean, op=MPI.SUM)
            else:
                mean = np.mean(data, axis=0)
            return mean
        if self.spatial:
            x = x.transpose(0, 2, 3, 1).reshape(-1, self.ci)

        if self.model.mode == TRAIN_MODE:
            N = np.array([x.shape[0]], dtype=self.dtype)
            if self.sync_stats and self.model.comm is not None:
                self.model.comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)

            mu = mean(x, N, self.model.comm)
            xc = (x - mu)
            var = mean(xc ** 2, N, self.model.comm)

            self.std = np.sqrt(var + self.epsilon)
            self.xn = xc / self.std
            y = self.gamma * self.xn + self.beta

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
            self.updated_running_var = True

        else:  # EVALUATE_MODE
            # Original numpy-based code
            # std = np.sqrt(self.running_var + self.epsilon)
            # xn = (x - self.running_mean) / self.std
            # y = self.gamma * xn + self.beta

            # If self.running_var was updated on training we need to recompute self.inv_std!
            if self.updated_running_var: 
                self.updated_running_var = False
                self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)

            y = bn_inference_cython(x, self.running_mean, self.inv_std, self.gamma, self.beta)
        
        if self.spatial:
            y = y.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)

        return y

    def backward(self, dy):
        if self.spatial:
            dy = dy.transpose(0, 2, 3, 1).reshape(-1, self.ci)

        N = dy.shape[0]
        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)

        if self.need_dx:
            dx = (self.gamma / (self.std * N)) * (N * dy - self.xn * self.dgamma - self.dbeta)
            dx = dx.astype(self.dtype)

            if self.spatial:
                dx = dx.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)
            return dx


class AdditionBlock(Layer):

    def __init__(self, *args):
        super(AdditionBlock, self).__init__()
        self.paths = []
        for p in args:
            self.paths.append(p)
        self.out_shapes = []
        self.is_block_layer = True

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        for p in self.paths:
            for i, l in enumerate(p):
                l.tracer = self.tracer
                l.dtype = self.dtype
                l.model = self.model
                l.batch_size = self.batch_size
                l.id = self.model.id + i + 1
                l.matmul = self.matmul

                l.initialize(prev_shape, need_dx)
                prev_shape = l.shape

                self.fwd_time += l.fwd_time
                self.bwd_time += l.bwd_time
                self.nparams += l.nparams

            self.out_shapes.append(prev_shape)
            prev_shape = self.prev_shape
            self.model.id += len(p)

        assert all([o == self.out_shapes[0] for o in self.out_shapes])
        self.shape = self.out_shapes[0]

    def show(self, attrs=""):
        print(
            f"|{self.id:^7d}|{(type(self).__name__ + ' (%d-path)' % len(self.paths)):^26s}|{'':9s}|{str(self.shape):^15s}|{'':19s}|{'':24s}|")
        for i, p in enumerate(self.paths):
            print(f"|{('Path %d' % i):^7s}|{'':^26s}|{'':9s}|{'':15s}|{'':19s}|{'':24s}|")
            for l in p: l.show()

    def update_weights(self, optimizer):
        for p in self.paths:
            for l in p:
                l.update_weights(optimizer)

    def reduce_weights_async(self):
        for p in self.paths:
            for l in p:
                l.reduce_weights_async()

    def wait_allreduce_async(self):
        for p in self.paths:
            for l in p:
                l.wait_allreduce_async()

    def reduce_weights_sync(self):
        for p in self.paths:
            for l in p:
                l.reduce_weights_sync()

    def forward(self, x):
        x = [x] * len(self.paths)
        for i, p in enumerate(self.paths):
            for l in p:
                self.tracer.emit_event(PYDTNN_MDL_EVENT, l.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
                x[i] = l.forward(x[i])
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i > 0:
                self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_ELTW_SUM)
                x[0] += x[i]
                self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return x[0]

    def backward(self, dy):
        dx = [dy] * len(self.paths)
        for i, p in enumerate(self.paths):
            for l in reversed(p):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, l.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx[i] = l.backward(dx[i])
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i > 0:
                self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_ELTW_SUM)
                dx[0] += dx[i]
                self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return dx[0]


class ConcatenationBlock(Layer):

    def __init__(self, *args):
        super(ConcatenationBlock, self).__init__()
        self.paths = []
        for p in args:
            self.paths.append(p)
        self.out_shapes = []
        # The next attributes will be initialized later
        self.out_co = self.idx_co = None
        self.is_block_layer = True

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        for p in self.paths:
            for i, l in enumerate(p):
                l.tracer = self.tracer
                l.dtype = self.dtype
                l.model = self.model
                l.batch_size = self.batch_size
                l.id = self.model.id + i + 1
                l.matmul = self.matmul
                l.initialize(prev_shape, need_dx)
                prev_shape = l.shape
                self.fwd_time += l.fwd_time
                self.bwd_time += l.bwd_time
                self.nparams += l.nparams

            self.out_shapes.append(prev_shape)
            prev_shape = self.prev_shape
            self.model.id += len(p)

        assert all([tuple(o[1:]) == tuple(self.out_shapes[0][1:]) for o in self.out_shapes])
        self.out_co = [s[0] for s in self.out_shapes]
        self.idx_co = np.cumsum(self.out_co, axis=0)
        self.shape = (sum(self.out_co), *self.out_shapes[0][1:])

    def show(self, attrs=""):
        print(
            f"|{self.id:^7d}|{(type(self).__name__.replace('Concatenation', 'Concat') + ' (%d-path)' % len(self.paths)):^26s}|{'':9s}|{str(self.shape):^15s}|{'':19s}|{'':24s}|")
        for i, p in enumerate(self.paths):
            print(f"|{('Path %d' % i):^7s}|{'':^26s}|{'':9s}|{'':15s}|{'':19s}|{'':24s}|")
            for l in p:
                l.show()

    def update_weights(self, optimizer):
        for p in self.paths:
            for l in p:
                l.update_weights(optimizer)

    def reduce_weights_async(self):
        for p in self.paths:
            for l in p:
                l.reduce_weights_async()

    def wait_allreduce_async(self):
        for p in self.paths:
            for l in p:
                l.wait_allreduce_async()

    def reduce_weights_sync(self):
        for p in self.paths:
            for l in p:
                l.reduce_weights_sync()

    def forward(self, x):
        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_REPLICATE)
        x = [x] * len(self.paths)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        for i, p in enumerate(self.paths):
            for l in p:
                self.tracer.emit_event(PYDTNN_MDL_EVENT, l.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
                x[i] = l.forward(x[i])
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONCAT)
        y = np.concatenate(x, axis=1)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def backward(self, dy):
        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SPLIT)
        dx = np.split(dy, self.idx_co[:-1], axis=1)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        for i, p in enumerate(self.paths):
            for l in reversed(p):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, l.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx[i] = l.backward(dx[i])
                self.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i > 0:
                self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_ELTW_SUM)
                dx[0] += dx[i]
                self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return dx[0]


class Conv2DRelu(Conv2D):

    def __init__(self, nfilters=1, filter_shape=(3, 3), padding=0, stride=1,
                 activation="", use_bias=True, weights_initializer="glorot_uniform",
                 biases_initializer="zeros", from_parent=None):
        if from_parent is None:
            super(Conv2DRelu, self).__init__(nfilters, filter_shape, padding, stride,
                 activation, use_bias, weights_initializer, biases_initializer)
        else:
            with suppress(KeyError):
                from_parent.__dict__.pop("forward")
            self.__dict__.update(from_parent.__dict__)

    def forward(self, x):
        """Version of the forward function that uses the convGemm + Relu"""

        if self.model.mode == TRAIN_MODE:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        # @todo: Replace ConvGemm by the actual fused layer
        res = self.cg.conv_gemm(self.weights, x, biases=None,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                biases_vector=biases_vector)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = res.reshape(self.co, -1, self.ho, self.wo).transpose(1, 0, 2, 3)
        self.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # @todo: Remove once ConvGemm+Relu is implemented !!
        y[y < 0] = 0

        return y


class BatchNormalizationRelu(BatchNormalization):

    def __init__(self, beta=0.0, gamma=1.0,
                 momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 sync_stats=False, from_parent=None):
        if from_parent is None:
            super(BatchNormalizationRelu, self).__init__(beta, gamma, momentum,
                epsilon, moving_mean_initializer, moving_variance_initializer, sync_stats)
        else:
            with suppress(KeyError):
                from_parent.__dict__.pop("forward")
            self.__dict__.update(from_parent.__dict__)

    def forward(self, x):
        """Version of the forward function that uses the BN + Relu"""

        if self.model.mode == TRAIN_MODE:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        if self.spatial:
            x = x.transpose(0, 2, 3, 1).reshape(-1, self.ci)

        y = bn_relu_inference_cython(x, self.running_mean, self.inv_std, self.gamma, self.beta)

        if self.spatial:
            y = y.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)

        return y