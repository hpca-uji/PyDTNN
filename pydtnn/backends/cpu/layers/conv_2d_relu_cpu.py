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

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Conv2DRelu
from pydtnn.model import TRAIN_MODE
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CONVGEMM, \
    PYDTNN_OPS_FORWARD_RESHAPE_Y


# Next no inspection due to Conv2D _backward_depthwise and _backward_pointwise being considered as abstract methods
# noinspection PyAbstractClass
from pydtnn.utils.best_transpose_1023 import best_transpose_1023


class Conv2DReluCPU(LayerCPU, Conv2DRelu):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape=None, need_dx=True, from_parent_dict=None):
        self.forward = {"_forward_nchw_cg": self._forward_nchw_cg, \
                        "_forward_nchw_cw": self._forward_nchw_cw}[from_parent_dict["forward"].__name__]
        self.weights = from_parent_dict["weights"]
        self.biases = from_parent_dict["biases"]

    def forward(self, x):
        """This is a fake forward function. It will be masked on initialization by a _forward implementation"""
        pass

    def _forward_nchw_cg(self, x):
        """Version of the forward function that uses the convGemm + Relu"""

        if self.model.mode == TRAIN_MODE:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        # @todo: Replace ConvGemm by the actual fused layer
        res = self.cg.conv_gemm_nchw(self.weights, x, biases=None,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation,
                                biases_vector=biases_vector)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = best_transpose_1023(res.reshape(self.co, -1, self.ho, self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # @todo: Remove once ConvGemm+Relu is implemented !!
        y[y < 0] = 0

        return y

    def _forward_nchw_cw(self, x):
        """Version of the forward function that uses the convWinograd + Relu"""

        if self.model.mode == TRAIN_MODE:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        y = self.cw.conv_winograd_nchw(self.weights, x, biases_vector,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation,
                                relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def backward(self, x):
        raise RuntimeError(f"Backward method of {self.__class__.__name__} should not be called")
