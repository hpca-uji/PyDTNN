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

import numpy as np
from pydtnn.backends.cpu.layers.abstract_pool_2d_layer_cpu import AbstractPool2DLayerCPU
from pydtnn.layers import AveragePool2D

from pydtnn.cython_modules import im2row_1ch_nhwc_cython, row2im_1ch_nhwc_cython, \
                                  average_pool_2d_fwd_nhwc_cython, average_pool_2d_bwd_nhwc_cython
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_COMP_DX_COL2IM, PYDTNN_OPS_FORWARD_IM2COL


class AveragePool2DCPU(AbstractPool2DLayerCPU, AveragePool2D):

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        setattr(self, "forward", self._forward_cython)
        setattr(self, "backward", self._backward_cython)
        # setattr(self, "forward", self._forward_i2c)
        # setattr(self, "backward", self._backward_i2c)

    def forward(self, x):
        """This is a fake forward function. It will be masked on initialization by _forward_i2c or _forward_cg"""
        pass

    def backward(self, dy):
        """This is a fake backward function. It will be masked on initialization by _backward_i2c or _backward_cg"""
        pass

    def _forward_i2c(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        x_rows = im2row_1ch_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        y = np.mean(x_rows, axis=1)
        return y.reshape(-1, self.ho, self.wo, self.co)

    def _forward_cython(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL)
        y = average_pool_2d_fwd_nhwc_cython(x, self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def _backward_i2c(self, dy):
        if self.need_dx:
            pool_size = np.prod(self.pool_shape)
            dy_rows = np.tile(dy.reshape(-1, 1) / pool_size, (1, pool_size))
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = row2im_1ch_nhwc_cython(dy_rows, dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            dx = dx.reshape(-1, self.hi, self.wi, self.ci)
            return dx

    def _backward_cython(self, dy):
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM)
            dx = average_pool_2d_bwd_nhwc_cython(dy, dy.shape[0], self.hi, self.wi, self.ci,
                                        self.kh, self.kw, self.vpadding, self.hpadding,
                                        self.vstride, self.hstride)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx
