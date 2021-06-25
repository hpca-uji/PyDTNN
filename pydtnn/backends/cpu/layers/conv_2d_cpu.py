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

from typing import Callable, List, Optional

import numpy as np

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.backends.cpu.libs import ConvGemm, ConvGemmCache
from pydtnn.cython_modules import im2row_nhwc_cython, add_nhwc_cython, row2im_nhwc_cython, \
    im2col_nchw_cython, add_nchw_cython, col2im_nchw_cython, transpose_1023_and_pad_cython, \
    reindex_cython, \
    depthwise_conv_cython
from pydtnn.layers import Conv2D
from pydtnn.model import TRAIN_MODE, EVALUATE_MODE
from pydtnn.performance_models import im2col_time, matmul_time, col2im_time
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CONVGEMM, \
    PYDTNN_OPS_FORWARD_RESHAPE_Y, \
    PYDTNN_OPS_COMP_DW_MATMUL, PYDTNN_OPS_COMP_DX_COL2IM, PYDTNN_OPS_COMP_DX_MATMUL, PYDTNN_OPS_FORWARD_IM2COL, \
    PYDTNN_OPS_BACKWARD_TRANSPOSE_DY, PYDTNN_OPS_BACKWARD_PADDING_X, \
    PYDTNN_OPS_BACKWARD_COMP_NEW_INDEXES, PYDTNN_OPS_BACKWARD_REINDEX, PYDTNN_OPS_BACKWARD_CONVGEMM, \
    PYDTNN_OPS_BACKWARD_SUM_BIASES, PYDTNN_OPS_FORWARD_MATMUL, PYDTNN_OPS_FORWARD_SUM_BIASES, \
    PYDTNN_OPS_FORWARD_RESHAPE_W, PYDTNN_OPS_BACKWARD_TRANSPOSE_W, PYDTNN_OPS_BACKWARD_RESHAPE_DW, \
    PYDTNN_OPS_BACKWARD_IM2COL, PYDTNN_OPS_BACKWARD_DECONV_GEMM, PYDTNN_OPS_FORWARD_DEPTHWISE_CONV, \
    PYDTNN_OPS_FORWARD_POINTWISE_CONV, PYDTNN_OPS_FORWARD_TRANSPOSE_Y
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW
from pydtnn.utils.best_of import BestOf
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from pydtnn.utils.best_transpose_1023 import best_transpose_1023


class Conv2DCPU(LayerCPU, Conv2D):
    _best_fw: Optional[BestOf] = None
    _best_fw_bw_pipeline: Optional[BestOf] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convGemm related attributes (some of them will be modified in initialize())
        self.cg = None
        self.cg_fallback_to_im2col = True  # Fallback to backward I2C if any stride is greater than 1
        self.cg_cache = True  # Store created matrices to allow them to be reused
        self.cg_deconv = False  # Use deconvGemm instead of matmul + col2im
        self.cg_trans = False  # Use the convGemm function to operate with im2colT
        self.cg_x_transposed_cache = ConvGemmCache(lambda shape: np.zeros(shape, self.model.dtype, order="C"))
        self.cg_x_indexed_cache = ConvGemmCache(lambda shape: np.zeros(shape, self.model.dtype, order="C"))
        self.cg_biases_cache = ConvGemmCache(lambda shape: np.empty(shape, self.model.dtype, order="F"))  # order F!
        self.cg_matmul_out_cache = ConvGemmCache(lambda shape: np.empty(shape, self.model.dtype, order="C"))

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        # Weights
        self.weights = self.weights_initializer(self.weights_shape, self.model.dtype)
        # Biases
        if self.use_bias:
            self.biases = self.biases_initializer((self.co,), self.model.dtype)
        # Set convGemm parameters
        if self.model.enable_conv_gemm:
            self.cg = ConvGemm(dtype=self.model.dtype, debug=self.debug, parent_layer=self)
            if not self.model.conv_gemm_cache:
                ConvGemmCache.disable()
            self.cg_fallback_to_im2col = self.model.conv_gemm_fallback_to_im2col
            self.cg_deconv = self.model.conv_gemm_deconv
            self.cg_trans = self.model.conv_gemm_trans
        # Set forward and backward implementations
        variant = 'i2c'  # Use i2c as default
        if self.grouping == "pointwise":
            variant = 'pointwise'
        elif self.grouping == "depthwise":
            variant = 'depthwise'
        elif self.model.enable_best_of:
            variant = 'best_of'
            if self.__class__._best_fw is None:
                self.__class__._best_fw = BestOf(
                    name="Conv2DCPU only forward",
                    alternatives=[
                        ('i2c', self._get_class_forward_and_backward('i2c')[0]),
                        ('cg', self._get_class_forward_and_backward('cg')[0]),
                    ],
                    get_problem_size=lambda *args: tuple(list(args[0].shape) + list(args[0].weights.shape)),
                )
                self.__class__._best_fw_bw_pipeline = BestOf(
                    name="Conv2DCPU forward backward",
                    alternatives=[
                        ('i2c', self._get_class_forward_and_backward('i2c')),
                        ('cg', self._get_class_forward_and_backward('cg')),
                    ],
                    get_problem_size=lambda *args: tuple(list(args[0].shape) + list(args[0].weights.shape)),
                )
            # Fix ConvGemm parameters to use convGemmTrans and Persistent memory (CGT+PM)
            if self.cg is None:
                self.cg = ConvGemm(dtype=self.model.dtype, debug=self.debug, parent_layer=self)
            ConvGemmCache.enable()
            self.cg_trans = True
            self.cg_fallback_to_im2col = False
            self.cg_deconv = False
        elif self.model.enable_conv_gemm:
            variant = 'cg'
        forward, backward = self._get_forward_and_backward(variant)
        setattr(self, "forward", forward)
        setattr(self, "backward", backward)
        # Performance models
        self.fwd_time = \
            im2col_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        if need_dx:
            self.bwd_time += matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                         k=self.co, cpu_speed=self.model.cpu_speed,
                                         memory_bw=self.model.memory_bw, dtype=self.model.dtype)
        else:
            self.bwd_time += col2im_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                         cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                                         dtype=self.model.dtype)

    def forward(self, x):
        """This is a fake forward function. It will be masked on initialization by a _forward implementation"""
        pass

    def backward(self, dy):
        """This is a fake backward function. It will be masked on initialization by a _backward implementation"""
        pass

    def _get_forward_and_backward(self, variant):
        tensor_format = 'nchw' if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW else 'nhwc'
        return (getattr(self, f'_forward_{tensor_format}_{variant}'),
                getattr(self, f'_backward_{tensor_format}_{variant}'))

    def _get_class_forward_and_backward(self, variant) -> List[Callable]:
        tensor_format = 'nchw' if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW else 'nhwc'
        return [getattr(self.__class__, f'_forward_{tensor_format}_{variant}'),
                getattr(self.__class__, f'_backward_{tensor_format}_{variant}')]

    def _forward_nhwc_i2c(self, x):
        """Version of the forward function that uses im2col and matmul"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_rows = im2row_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.model.mode == TRAIN_MODE:
            self.x_rows = x_rows

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_W)
        w_cols = self.weights.reshape(-1, self.co)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_MATMUL)
        res = self.model.matmul(x_rows, w_cols)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_SUM_BIASES)
        y = add_nhwc_cython(res, self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = y.reshape(-1, self.ho, self.wo, self.co)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def _forward_nhwc_cg(self, x):
        raise NotImplementedError("Forward not yet implemented!")

    def _forward_nhwc_depthwise(self, x):
        raise NotImplementedError("Forward not yet implemented!")

    def _forward_nhwc_pointwise(self, x):
        raise NotImplementedError("Forward not yet implemented!")

    def _fw_bw_best_of(self, stage, x_or_y):
        if self.model.mode == TRAIN_MODE:
            # noinspection PyTypeChecker
            return self._best_fw_bw_pipeline(stage, self, x_or_y)
        elif self.model.mode == EVALUATE_MODE:
            # noinspection PyTypeChecker
            return self._best_fw(self, x_or_y)
        else:
            raise RuntimeError("Conv2D BestOf variant requires to Model.mode to be set to EVALUATE_MODE or TRAIN_MODE")

    def _forward_nhwc_best_of(self, x):
        return self._fw_bw_best_of(0, x)

    def _forward_nchw_i2c(self, x):
        """Version of the forward function that uses im2col and matmul"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_cols = im2col_nchw_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.model.mode == TRAIN_MODE:
            self.x_cols = x_cols

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_W)
        w_cols = self.weights.reshape(self.co, -1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_MATMUL)
        res = self.model.matmul(w_cols, x_cols)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_SUM_BIASES)
        y = add_nchw_cython(res, self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = best_transpose_1023(y.reshape(self.co, -1, self.ho, self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def _forward_nchw_cg(self, x):
        """Version of the forward function that uses the convGemm library"""

        if self.model.mode == TRAIN_MODE:
            self.cg_x = x

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        res = self.cg.conv_gemm(self.weights, x, biases=None,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation,
                                biases_vector=biases_vector)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # Biases sum is now done on conv_gemm
        # y = res + self.biases.reshape(-1, 1)
        # y = add_cython(res, self.biases) if self.use_bias else res

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = best_transpose_1023(res.reshape(self.co, -1, self.ho, self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def _forward_nchw_depthwise(self, x):
        """ Version of the forward that perform a depthwise convolution"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_DEPTHWISE_CONV)
        res = depthwise_conv_cython(x, self.weights, self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_SUM_BIASES)
        y = add_nchw_cython(res, self.biases) if self.use_bias else res
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = best_transpose_1023(y.reshape(self.co, -1, self.ho, self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def _forward_nchw_pointwise(self, x):

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_POINTWISE_CONV)
        # y = np.einsum("nchw,oc->nohw", x, self.weights) # Einsum
        y = np.matmul(best_transpose_0231(x), np.transpose(self.weights, axes=(1, 0)))  # Matmul
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_TRANSPOSE_Y)
        y = best_transpose_0312(y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_SUM_BIASES)
        if self.use_bias:
            y += self.biases.reshape(1, self.co, 1, 1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def _forward_nchw_best_of(self, x):
        return self._fw_bw_best_of(0, x)

    def _backward_nhwc_i2c(self, dy):
        """Version of the backward function that uses im2col and matmul"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_DY)
        dy_rows = dy.reshape(-1, self.co)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DW_MATMUL)
        res = self.model.matmul(self.x_rows.T, dy_rows)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_RESHAPE_DW)
        self.dw = res.reshape(self.weights.shape)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 1, 2))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_W)
            w_rows = self.weights.reshape(-1, self.co)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL)
            res = self.model.matmul(dy_rows, w_rows.T)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = row2im_nhwc_cython(res, dy.shape[0], self.hi, self.wi, self.ci,
                                    self.kh, self.kw, self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx

    def _backward_nhwc_cg(self, dy):
        raise NotImplementedError("Backward not yet implemented!")

    def _backward_nhwc_depthwise(self, dy):
        raise NotImplementedError("Backward not yet implemented!")

    def _backward_nhwc_pointwise(self, dy):
        raise NotImplementedError("Backward not yet implemented!")

    def _backward_nhwc_best_of(self, y):
        return self._fw_bw_best_of(1, y)

    def _backward_nchw_i2c(self, dy):
        """Version of the backward function that uses im2col and matmul"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_DY)
        dy_cols = best_transpose_1023(dy).reshape(self.co, -1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DW_MATMUL)
        res = self.model.matmul(dy_cols, self.x_cols.T)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_RESHAPE_DW)
        self.dw = res.reshape(self.weights.shape)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 2, 3))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_W)
            w_cols = self.weights.reshape(self.co, -1).T
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL)
            res = self.model.matmul(w_cols, dy_cols)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = col2im_nchw_cython(res, dy.shape[0], self.ci, self.hi, self.wi,
                                    self.kh, self.kw, self.vpadding, self.hpadding,
                                    self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx

    @staticmethod
    # @lru_cache(maxsize=4)
    def _get_x_new_indexes_and_xstride(kx, xo, s, d):
        """
        Returns x_reorder and xstride based on kx (kh or kw), xo (ho or wo), and s (hstride or
        vstride)
        """
        if s == 1:
            return None, 1
        x_reorder = []
        for i in range(kx):
            x_reorder += [i * d + j * s for j in range(xo)]
        # Return x_reorder as a numpy.array because indexing is faster with a numpy.array than with a list
        return np.array(x_reorder), xo

    def _backward_nchw_cg(self, dy):
        """Version of the backward function that uses the convGemm library"""

        # if self.id == 4:
        #     self.model.tracer.print_memory_usage(f"Inside layer {self.id:03} backward 01")

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
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_IM2COL)
            self.x_cols = im2col_nchw_cython(self.cg_x, self.kh, self.kw, self.vpadding, self.hpadding,
                                             self.vstride, self.hstride, self.vdilation, self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return self._backward_nchw_i2c(dy)

        if not self.cg_trans:
            # 1) cg_dy
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_DY)
            cg_dy = best_transpose_1023(dy)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 2) cg_x_transposed
            b, c, h, w = self.cg_x.shape
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_PADDING_X)
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
            #     # self.cg_x_indexed = np.zeros((c, b, new_h, new_w), self.model.dtype)
            #     self.cg_x_indexed = self.cg_x_indexed_cache[(c, b, new_h, new_w)]
            #     self.cg_x_indexed[:, :, self.vpadding:new_h - self.vpadding, self.hpadding:new_w - self.hpadding] = \
            #         self.cg_x.transpose((1, 0, 2, 3))
            if self.vpadding == 0 and self.hpadding == 0:
                cg_x_transposed = best_transpose_1023(self.cg_x)
            else:
                new_h, new_w = h + 2 * self.vpadding, w + 2 * self.hpadding
                cg_x_transposed = self.cg_x_transposed_cache[(c, b, new_h, new_w)]
                transpose_1023_and_pad_cython(self.cg_x, cg_x_transposed)
            cg_vpadding = 0
            cg_hpadding = 0
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 3) new indexes and strides
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_COMP_NEW_INDEXES)
            v_new_indexes, cg_vstride = self._get_x_new_indexes_and_xstride(self.kh, self.ho, self.vstride,
                                                                            self.vdilation)
            h_new_indexes, cg_hstride = self._get_x_new_indexes_and_xstride(self.kw, self.wo, self.hstride,
                                                                            self.hdilation)
            cg_vdilation = self.vdilation
            cg_hdilation = self.hdilation
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 4) cg_x_indexed
            # Indexing performance considerations:
            #  + Using numpy.array to select the indexes is faster than using a list
            #    (check _get_x_new_indexes_and_xstride).
            #  + Indexing first rows and then columns is faster than the opposite.
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_REINDEX)
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
                # self.cg_x_indexed = np.empty((c, b, new_h, new_w), dtype=self.model.dtype)
                cg_x_indexed = self.cg_x_indexed_cache[(c, b, new_h, new_w)]
                reindex_cython(v_new_indexes, h_new_indexes, cg_x_transposed, cg_x_indexed)
            else:
                cg_x_indexed = cg_x_transposed
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 5) cg_biases
            cg_biases = None
        else:
            # if self.cg_trans:
            # 1) cg_dy
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_TRANSPOSE_DY)
            cg_dy = best_transpose_1023(dy)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            # 2) cg_x_indexed
            cg_x_indexed = self.cg_x
            # 3) cg_biases
            cg_biases = self.cg_biases_cache[self.weights.shape]
            # 4) rest of parameters
            cg_vpadding = self.vpadding
            cg_hpadding = self.hpadding
            cg_vstride = self.vstride
            cg_hstride = self.hstride
            cg_vdilation = self.vdilation
            cg_hdilation = self.hdilation

        if self.debug:
            # Expose cg_x_indexed
            self.cg_x_indexed = cg_x_indexed

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CONVGEMM)
        res = self.cg.conv_gemm(cg_dy, cg_x_indexed,
                                biases=cg_biases, beta=0.0,
                                vpadding=cg_vpadding, hpadding=cg_hpadding,
                                vstride=cg_vstride, hstride=cg_hstride,
                                vdilation=cg_vdilation, hdilation=cg_hdilation, trans=self.cg_trans)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_RESHAPE_DW)
        self.dw = res.reshape(self.weights.shape)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 2, 3))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.need_dx:
            if self.cg_deconv:
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_DECONV_GEMM)
                dx = self.cg.deconv_gemm(self.weights, dy, self.cg_x,
                                         vpadding=self.vpadding, hpadding=self.hpadding,
                                         vstride=self.vstride, hstride=self.hstride,
                                         vdilation=self.vdilation, hdilation=self.hdilation)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            else:
                w_cols = self.weights.reshape(self.co, -1).T
                dy_cols = cg_dy.reshape(self.co, -1)

                # Don't use a cached version if the res matrix will persist
                res = self.cg_matmul_out_cache[(w_cols.shape[0], dy_cols.shape[1])]

                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL)
                self.model.matmul(w_cols, dy_cols, res)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
                dx = col2im_nchw_cython(res, dy.shape[0], self.ci, self.hi, self.wi,
                                        self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride, self.vdilation, self.hdilation)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx

    def _backward_nchw_depthwise(self, dy):
        raise NotImplementedError("Backward not yet implemented!")

    def _backward_nchw_pointwise(self, dy):
        raise NotImplementedError("Backward not yet implemented!")

    def _backward_nchw_best_of(self, y):
        return self._fw_bw_best_of(1, y)
