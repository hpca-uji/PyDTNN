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
from pydtnn.backends.cpu.libs import ConvGemm, ConvWinograd, is_conv_gemm_available, is_conv_winograd_available
from pydtnn.cython_modules import im2row_nhwc_cython, add_nhwc_cython, row2im_nhwc_cython, \
    im2col_nchw_cython, add_nchw_cython, col2im_nchw_cython, \
    depthwise_conv_cython
from pydtnn.layers import Conv2D
from pydtnn.model import TRAIN_MODE, EVALUATE_MODE
from pydtnn.performance_models import im2col_time, matmul_time, col2im_time
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CONVGEMM, \
    PYDTNN_OPS_FORWARD_RESHAPE_Y, \
    PYDTNN_OPS_COMP_DW_MATMUL, PYDTNN_OPS_COMP_DX_COL2IM, PYDTNN_OPS_COMP_DX_MATMUL, PYDTNN_OPS_FORWARD_IM2COL, \
    PYDTNN_OPS_BACKWARD_TRANSPOSE_DY, \
    PYDTNN_OPS_BACKWARD_CONVGEMM, \
    PYDTNN_OPS_BACKWARD_SUM_BIASES, PYDTNN_OPS_FORWARD_MATMUL, PYDTNN_OPS_FORWARD_SUM_BIASES, \
    PYDTNN_OPS_FORWARD_RESHAPE_W, PYDTNN_OPS_BACKWARD_TRANSPOSE_W, PYDTNN_OPS_BACKWARD_RESHAPE_DW, \
    PYDTNN_OPS_BACKWARD_IM2COL, PYDTNN_OPS_BACKWARD_DECONV_GEMM, PYDTNN_OPS_FORWARD_DEPTHWISE_CONV, \
    PYDTNN_OPS_FORWARD_POINTWISE_CONV, PYDTNN_OPS_FORWARD_TRANSPOSE_Y, PYDTNN_OPS_FORWARD_CONVWINOGRAD
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW
from pydtnn.utils.memory_cache import MemoryCache
from pydtnn.utils.best_of import BestOf
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from pydtnn.utils.best_transpose_1023 import best_transpose_1023


class Conv2DCPU(LayerCPU, Conv2D):
    # _best_fw: Optional[BestOf] = None
    # _best_fw_bw_pipeline: Optional[BestOf] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convGemm related attributes (some of them will be modified in initialize())
        self.cg = None
        # convWinograd related attributes (some of them will be modified in initialize())
        self.cw = None
        self._best_fw = None
        self._best_fw_bw_pipeline = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        # Weights
        self.weights = self.weights_initializer(self.weights_shape, self.model.dtype)
        # Biases
        if self.use_bias:
            self.biases = self.biases_initializer((self.co,), self.model.dtype)
        if not self.model.enable_memory_cache:
            MemoryCache.disable()
        cw_constraints_fulfilled = None
        # Set convWinograd parameters
        if self.model.enable_conv_winograd:
            try:
                self.cw = ConvWinograd(self.kh, self.kw, self.vstride, self.hstride,
                                       self.vdilation, self.hdilation,
                                       dtype=self.model.dtype, tensor_format=self.model.tensor_format,
                                       debug=self.debug, parent_layer=self)
                cw_constraints_fulfilled = True
            except NotImplementedError:
                cw_constraints_fulfilled = False
        # Set convGemm parameters
        if self.model.enable_conv_gemm:
            self.cg = ConvGemm(dtype=self.model.dtype, debug=self.debug, parent_layer=self)
        # Set forward and backward implementations
        variant = 'i2c'  # Use i2c as default
        if self.grouping == "pointwise":
            variant = 'pointwise'
        elif self.grouping == "depthwise":
            variant = 'depthwise'
        elif self.model.enable_best_of:
            variant = 'best_of'

            # Fix ConvGemm parameters to use convGemmTrans and Persistent memory (CGT+PM)
            MemoryCache.enable()
            if is_conv_winograd_available and self.cw is None and cw_constraints_fulfilled is None:
                try:
                    self.cw = ConvWinograd(self.kh, self.kw, self.vstride, self.hstride,
                                           self.vdilation, self.hdilation,
                                           dtype=self.model.dtype, tensor_format=self.model.tensor_format,
                                           debug=self.debug, parent_layer=self)
                    cw_constraints_fulfilled = True
                except NotImplementedError:
                    cw_constraints_fulfilled = False
            if is_conv_gemm_available:
                if self.cg is None:
                    self.cg = ConvGemm(dtype=self.model.dtype, debug=self.debug, parent_layer=self)

            # if self.__class__._best_fw is None:
            alternatives_fw=[ ('i2c', self._get_class_forward_and_backward('i2c')[0]) ]
            if is_conv_gemm_available:
                alternatives_fw.append( ('cg', self._get_class_forward_and_backward('cg')[0]) )
            if is_conv_winograd_available and cw_constraints_fulfilled:
                alternatives_fw.append( ('cw', self._get_class_forward_and_backward('cw')[0]) )

            self._best_fw = BestOf(
                name="Conv2DCPU only forward",
                alternatives=alternatives_fw,
                get_problem_size=lambda *args: tuple(list(args[0].shape) + list(args[0].weights.shape)),
            )

            alternatives_fw_bw_pipeline=[ ('i2c', self._get_class_forward_and_backward('i2c')) ]
            if is_conv_gemm_available:
                alternatives_fw_bw_pipeline.append( ('cg', self._get_class_forward_and_backward('cg')) )
            if is_conv_winograd_available and cw_constraints_fulfilled:
                alternatives_fw_bw_pipeline.append( ('cw', self._get_class_forward_and_backward('cw')) )

            self._best_fw_bw_pipeline = BestOf(
                name="Conv2DCPU forward backward",
                alternatives=alternatives_fw_bw_pipeline,
                get_problem_size=lambda *args: tuple(list(args[0].shape) + list(args[0].weights.shape)),
            )
        elif self.model.enable_conv_winograd:
            if cw_constraints_fulfilled:
                MemoryCache.enable()
                variant = 'cw'
            elif self.model.enable_conv_gemm:
                MemoryCache.enable()
                variant = 'cg'
            else:
                variant = 'i2c'
        elif self.model.enable_conv_gemm:
            variant = 'cg'
        self.cw_constraints_fulfilled = cw_constraints_fulfilled
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
        """Version of the forward function that uses the convGemm library"""

        if self.model.mode == TRAIN_MODE:
            self.cg_x = x

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        y = self.cg.conv_gemm_nhwc(self.weights, x,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation,
                                biases_vector=biases_vector)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def _forward_nhwc_cw(self, x):
        """Version of the forward function that uses the convWinograd library"""

        if self.model.mode == TRAIN_MODE:
            self.cw_x = x

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVWINOGRAD)
        y = self.cw.conv_winograd_nhwc(self.weights, x, biases=biases_vector,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

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
        res = self.cg.conv_gemm_nchw(self.weights, x, biases=None,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation,
                                biases_vector=biases_vector)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return res

    def _forward_nchw_cw(self, x):
        """Version of the forward function that uses the convWinograd library"""

        if self.model.mode == TRAIN_MODE:
            self.cw_x = x

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVWINOGRAD)
        y = self.cw.conv_winograd_nchw(self.weights, x, biases=biases_vector,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation)
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

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CONVGEMM)
        res = np.empty(self.weights.shape, dtype=dy.dtype)
        self.cg.conv_gemm_nhwc(dy, self.cg_x, biases=res,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation,
                                trans=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        self.dw = res

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 1, 2))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.need_dx:

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_DECONV_GEMM)
            dx = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=dy.dtype)
            self.cg.deconv_gemm_nhwc(self.weights, dy, dx,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            return dx

    def _backward_nhwc_cw(self, dy):
        return self._backward_nhwc_i2c(dy)

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

    def _backward_nchw_cg(self, dy):
        """Version of the backward function that uses the convGemm library"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CONVGEMM)
        res = np.empty(self.weights.shape, dtype=dy.dtype)
        self.cg.conv_gemm_nchw(dy, self.cg_x, biases=res,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation,
                                trans=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        self.dw = res

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 2, 3))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_DECONV_GEMM)
            dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=dy.dtype)
            self.cg.deconv_gemm_nchw(self.weights, dy, dx,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            return dx

    def _backward_nchw_cw(self, dy):
        """Version of the backward function that uses the convWinograd library"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_IM2COL)
        self.x_cols = im2col_nchw_cython(self.cw_x, self.kh, self.kw, self.vpadding, self.hpadding,
                                         self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self._backward_nchw_i2c(dy)

    def _backward_nchw_depthwise(self, dy):
        raise NotImplementedError("Backward not yet implemented!")

    def _backward_nchw_pointwise(self, dy):
        raise NotImplementedError("Backward not yet implemented!")

    def _backward_nchw_best_of(self, y):
        return self._fw_bw_best_of(1, y)
