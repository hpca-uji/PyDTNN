#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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

from pydtnn.backends.cpu.layers.conv_2d_variants.i2c_variant import I2CVariant
from pydtnn.backends.cpu.libs import ConvWinograd
from pydtnn.cython_modules import im2row_nhwc_cython, im2col_nchw_cython
from pydtnn.model import ModelModeEnum
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from numpy import ndarray, zeros
class ConvWinogradVariant(I2CVariant, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convWinograd related attributes (will be initialized in initialize())
        self.cw = None
        self.cw_constraints_fulfilled = None

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        # ConvWinograd parameters
        if self.model.enable_conv_winograd:
            try:
                self.cw = ConvWinograd(self.kh, self.kw, self.vstride, self.hstride,
                                       self.vdilation, self.hdilation,
                                       dtype=self.model.dtype, tensor_format=self.model.tensor_format,
                                       debug=self.debug, parent_layer=self)
            except NotImplementedError:
                self.cw_constraints_fulfilled = False
            else:
                self.cw_constraints_fulfilled = True

    def _forward_cw_nhwc(self, x:ndarray) -> ndarray:
        """Version of the forward function that uses the convWinograd library"""

        if self.model.mode is ModelModeEnum.TRAIN:
            self.cw_x = x

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVWINOGRAD)
        y:ndarray = self.cw.conv_winograd_nhwc(self.weights, x, biases=biases_vector,
                                               vpadding=self.vpadding, hpadding=self.hpadding,
                                               vstride=self.vstride, hstride=self.hstride,
                                               vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_cw_nchw(self, x:ndarray) -> ndarray:
        """Version of the forward function that uses the convWinograd library"""

        if self.model.mode is ModelModeEnum.TRAIN:
            self.cw_x = x

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVWINOGRAD)
        y:ndarray = self.cw.conv_winograd_nchw(self.weights, x, biases=biases_vector,
                                               vpadding=self.vpadding, hpadding=self.hpadding,
                                               vstride=self.vstride, hstride=self.hstride,
                                               vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_cw_nhwc(self, dy: ndarray) -> ndarray:
        """Version of the backward function that uses the convWinograd library"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_IM2COL)
        
        self.x_rows = zeros(((dy.shape[0] * self.ho * self.wo), (self.ci * self.kh * self.kw)), dtype=self.model.dtype)
        im2row_nhwc_cython(self.cw_x, self.x_rows,
                           self.kh, self.kw, self.ho, self.wo,
                           self.vpadding, self.hpadding,
                           self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self._backward_i2c_nhwc(dy)

    def _backward_cw_nchw(self, dy: ndarray) -> ndarray:
        """Version of the backward function that uses the convWinograd library"""
        n, c, _, _ = dy.shape
        self.x_cols = zeros((c * self.kh * self.kw, n * self.ho * self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_IM2COL)
        im2col_nchw_cython(self.cw_x, 
                           self.kh, self.kw, self.vpadding, self.hpadding,
                           self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self._backward_i2c_nchw(dy)
