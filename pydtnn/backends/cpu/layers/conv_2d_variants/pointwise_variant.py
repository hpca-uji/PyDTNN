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

from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_POINTWISE_CONV, \
    PYDTNN_OPS_FORWARD_TRANSPOSE_Y, PYDTNN_OPS_FORWARD_SUM_BIASES
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312


class PointwiseVariant(Conv2D, ABC):

    def _forward_pointwise_nhwc(self, x):
        raise RuntimeError("Forward not yet implemented!")

    def _forward_pointwise_nchw(self, x):
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

    def _backward_pointwise_nhwc(self, dy):
        raise RuntimeError("Backward not yet implemented!")

    def _backward_pointwise_nchw(self, dy):
        raise RuntimeError("Backward not yet implemented!")
