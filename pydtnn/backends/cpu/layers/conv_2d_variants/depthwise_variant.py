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

from pydtnn.cython_modules import depthwise_conv_cython, add_nchw_cython
from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_DEPTHWISE_CONV, \
    PYDTNN_OPS_FORWARD_SUM_BIASES, PYDTNN_OPS_FORWARD_RESHAPE_Y
from pydtnn.utils.best_transpose_1023 import best_transpose_1023


class DepthwiseVariant(Conv2D, ABC):

    def _forward_depthwise_nhwc(self, x):
        raise RuntimeError("Forward not yet implemented!")

    def _forward_depthwise_nchw(self, x):
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

    def _backward_depthwise_nhwc(self, dy):
        raise RuntimeError("Backward not yet implemented!")

    def _backward_depthwise_nchw(self, dy):
        raise RuntimeError("Backward not yet implemented!")
