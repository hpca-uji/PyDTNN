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

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel

from pydtnn import layers
from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_RESHAPE_Y, PYDTNN_OPS_BACKWARD_RESHAPE_DX
from .layer_gpu_mixin import LayerGPUMixin
from ..tensor_gpu import TensorGPU


class FlattenGPU(LayerGPUMixin, layers.Flatten):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        self.y = x

    def forward(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = x.reshape((self.model.batch_size, np.prod(self.prev_shape)))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def backward(self, dy):
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_RESHAPE_DX)
            dx = dy.reshape((self.model.batch_size, *self.prev_shape))
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx
