#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

from abc import ABC

import numpy as np

from pydtnn.backends.cpu.libs import ConvGemm
from pydtnn.layers import Conv2D
from pydtnn.model import TRAIN_MODE
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CONVGEMM, \
    PYDTNN_OPS_BACKWARD_CONVGEMM, PYDTNN_OPS_BACKWARD_SUM_BIASES, PYDTNN_OPS_BACKWARD_DECONV_GEMM


class ConvGemmVariant(Conv2D, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convGemm related attributes (will be initialized in initialize())
        self.cg = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        # ConvGemm parameters
        if self.model.enable_conv_gemm:
            self.cg = ConvGemm(dtype=self.model.dtype, debug=self.debug, parent_layer=self)

    def _forward_cg_nhwc(self, x):
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

    def _forward_cg_nchw(self, x):
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

    def _backward_cg_nhwc(self, dy):
        """Version of the backward function that uses the convGemm library"""
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

    def _backward_cg_nchw(self, dy):
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
