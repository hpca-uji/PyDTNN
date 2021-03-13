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

from .layer_gpu import LayerGPU
from .. import layers
from ..performance_models import *
from ..tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_COPY, PYDTNN_OPS_BACKWARD_COPY
from ..utils import TensorGPU

try:
    # noinspection PyUnresolvedReferences
    from pycuda.elementwise import ElementwiseKernel
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
except (ImportError, ModuleNotFoundError):
    pass


class FlattenGPU(LayerGPU, layers.Flatten):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        self.copy = ElementwiseKernel("T *dst, T *src".replace("T",
                                                               {np.float32: "float",
                                                                np.float64: "double"}[self.model.dtype]),
                                      "dst[i] = src[i];", "copy")
        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, np.prod(prev_shape)), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)
        # Derivative dx
        if self.need_dx:
            dx_gpu = gpuarray.empty((self.model.batch_size, *prev_shape), self.model.dtype)
            self.dx = TensorGPU(dx_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)

    def forward(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_COPY)
        self.copy(self.y.ary, x.ary, stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_COPY)
            # Compute dx
            self.copy(self.dx.ary, dy.ary, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return self.dx
