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
from ..tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CUDNN, PYDTNN_OPS_BACKWARD_CUDNN_DX
from ..utils import TensorGPU

try:
    # noinspection PyUnresolvedReferences
    import libcudnn.libcudnn as cudnn
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
except (ImportError, ModuleNotFoundError):
    pass


class DropoutGPU(LayerGPU, layers.Dropout):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states_size = None
        self.space_size = None
        self.space = None
        self.states = None
        self.drop_desc = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)

        # Derivative dx
        if need_dx:
            dx_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
            self.dx = TensorGPU(dx_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)

        self.states_size = cudnn.cudnnDropoutGetStatesSize(self.model.cudnn_handle)
        self.space_size = cudnn.cudnnDropoutGetReserveSpaceSize(self.y.desc)

        space_gpu = gpuarray.empty((self.space_size.value,), np.byte)
        self.space = TensorGPU(space_gpu, self.model.tensor_fmt, self.model.cudnn_dtype, "other")

        states_gpu = gpuarray.empty((self.states_size.value,), np.byte)
        self.states = TensorGPU(states_gpu, self.model.tensor_fmt, self.model.cudnn_dtype, "other")

        self.drop_desc = cudnn.cudnnCreateDropoutDescriptor()

        cudnn.cudnnSetDropoutDescriptor(self.drop_desc, self.model.cudnn_handle, self.rate,
                                        self.states.ptr, self.states_size.value, seed=0)

    def forward(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN)
        cudnn.cudnnDropoutForward(self.model.cudnn_handle, self.drop_desc,
                                  x.desc, x.ptr,
                                  self.y.desc, self.y.ptr,
                                  self.space.ptr, self.space_size.value)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUDNN_DX)
            # Compute dx
            cudnn.cudnnDropoutBackward(self.model.cudnn_handle, self.drop_desc,
                                       dy.desc, dy.ptr,
                                       self.dx.desc, self.dx.ptr,
                                       self.space.ptr, self.space_size.value)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return self.dx
